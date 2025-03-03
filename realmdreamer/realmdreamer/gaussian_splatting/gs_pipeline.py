import json
import os
import pdb
import sys
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import (Any, Dict, List, Literal, Mapping, Optional, Tuple, Type,
                    Union, cast)

import cv2
import kornia
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.pipelines.base_pipeline import (Pipeline, VanillaPipeline,
                                                VanillaPipelineConfig)
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms
from tqdm.auto import tqdm
from typing_extensions import Literal
from utils.depth import (depth_pearson_loss, depth_ranking_loss,
                         depth_ranking_loss_multi_patch)

from .gs_sds_datamanager import GaussianSplattingDatamanagerConfig


@dataclass
class GaussianSplattingPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussianSplattingPipeline)
    """target class to instantiate"""

    datamanager: GaussianSplattingDatamanagerConfig = GaussianSplattingDatamanagerConfig()
    """specifies the datamanager config"""

    prompt: str = "beautiful"
    """prompt for SDS"""

    guidance_eval_steps: int = 250
    """Interval of steps evaluate the guidance"""

    # Gaussian splatting densification and pruning configs
    density_start_iter: int = 0
    """Start densification at this iteration"""

    density_end_iter: int = 999999999
    """End densification at this iteration"""

    densification_interval: int = 1000
    """Densify and prune every this many iterations"""

    densify_grad_threshold: float = 0.01
    """Densify if the gradient is above this threshold"""

    enable_prune: bool = True
    """Enable pruning"""

    opacity_reset_interval: int = 200
    """Reset opacity every this many iterations"""

    input_view_constraint: bool = False
    """Render the first view every step and add an anchor loss"""

    input_view_depth_constraint: bool = False
    """Render the first view every step and add an depth correlation loss"""

    input_view_depth_constraint_type: Literal["pearson", "ranking", "ranking_multi_patch"] = "pearson"
    """Type of depth correlation loss"""

    ## Leave as is:
    max_num_iterations: int = -1
    """Max number of iterations - value is set in the trainer - ignore this"""


class GaussianSplattingPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: GaussianSplattingPipelineConfig

    def __init__(
        self,
        config: GaussianSplattingPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        grad_scaler=None,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.grad_norms = []

        if self.config.input_view_constraint:
            self.lpips_loss = lpips.LPIPS(net="vgg").to("cuda")

    def get_train_loss_dict(self, step: int, get_image_dict: bool = False):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
            get_image_dict: whether to return the image dict or not
        """

        batch_start = time.time()
        train_batch = self.datamanager.next_train(step)
        batch_end = time.time()

        datafetch_time = batch_end - batch_start
        # print("Time to get batch: ", batch_end - batch_start)

        img_size = train_batch["img_size"]
        batch = train_batch["batch"]
        c2w = train_batch["c2w"]
        camera = train_batch["camera"]

        batch["inpainting_mask"] = batch["inpainting_mask"].unsqueeze(-1)
        batch["inpainting_mask_2"] = batch["inpainting_mask_2"].unsqueeze(-1)

        num_images = c2w.shape[0]

        model_start = time.time()
        # Render from the camera poses specified
        model_outputs = self.model(camera)
        model_outputs["rgb"] = model_outputs["rgb"].clamp(0, 1)
        model_end = time.time()

        render_time = model_end - model_start
        # print("Time to run model: ", model_end - model_start)

        metrics_dict = {}
        image_dict = {}

        # Compute the loss
        start_loss = time.time()
        loss_dict, misc = self.model.get_loss_dict(
            outputs=model_outputs,
            batch=batch,
            prompt=self.config.prompt,
            c2w=c2w,
            step_ratio=step / self.config.max_num_iterations,
        )
        end_loss = time.time()

        loss_time = end_loss - start_loss
        # print("Time to compute loss: ", end_loss - start_loss)

        gaussian_stats = self.model.get_stats()

        image_dict.update(
            {
                "Render": model_outputs["rgb"][0].float().permute(2, 0, 1),
                "Target Image": batch["image"].permute(0, 3, 1, 2)[0].clamp(0, 1),
                "Target Depth": batch["depth_image"].permute(0, 3, 1, 2)[0],
                "Inpainting Mask": batch["inpainting_mask"].permute(0, 3, 1, 2)[0].float(),
                "Inpainting Mask 2": batch["inpainting_mask_2"].permute(0, 3, 1, 2)[0].float(),
                "Masked Render": (model_outputs["rgb"][0].float().permute(2, 0, 1))
                * batch["inpainting_mask"].permute(0, 3, 1, 2)[0].float(),
                "Inverse Masked Render": (model_outputs["rgb"][0].float().permute(2, 0, 1))
                * (1 - batch["inpainting_mask"].permute(0, 3, 1, 2)[0].float()),
            }
        )

        if "one_step_pred" in misc.keys():

            image_dict.update({"One Step Pred": misc["one_step_pred"][0].float()})

        if "multi_step_pred" in misc.keys():

            image_dict.update({"Multi Step Pred": misc["multi_step_pred"][0].float()})

        # Test inpainting
        if step % self.config.guidance_eval_steps == 0 and get_image_dict:

            if "sds_inpainting" in self.model.config.guidance:

                inpainted_image, one_step = self.model.guidance.sample(
                    rgb=model_outputs["rgb"],  # Shape (B, H, W, 3)
                    prompt=self.config.prompt,
                    mask=batch["inpainting_mask"].to(self.device),  # Shape (B, H, W, 1)
                    strength=(
                        1 - step / self.config.max_num_iterations
                        if self.model.config.anneal
                        else torch.rand(1).item() * 0.88 + 0.1
                    ),
                )

                image_dict.update(
                    {
                        "Guidance Eval": inpainted_image[0],
                    }
                )

            elif "sds" in self.model.config.guidance:

                completed_image, one_step = self.model.guidance.sample(
                    rgb=model_outputs["rgb"],  # Shape (B, H, W, 3)
                    prompt=self.config.prompt,
                    mask=batch["inpainting_mask"].to(self.device),  # Shape (B, H, W, 1)
                    strength=(
                        1 - step / self.config.max_num_iterations
                        if self.model.config.anneal
                        else torch.rand(1).item()
                        * (self.model.config.max_step_percent - self.model.config.min_step_percent)
                        + self.model.config.min_step_percent
                    ),
                )

                image_dict.update(
                    {
                        "Guidance Eval": completed_image[0],
                    }
                )

        # use matplot to log
        if get_image_dict:

            self.grad_norms.append(misc["grad"].norm(dim=1, keepdim=True).detach().cpu().numpy()[0, 0])

            # Save the grad norm array
            # print("Saving to", os.path.join(wandb.run.dir, 'grad_norms.npy'))
            # np.save(os.path.join(wandb.run.dir, 'grad_norms.npy'), np.array(self.grad_norms))

            # Log the gradient of the loss with respect to the image
            fig = plt.figure()
            plt.imshow(
                misc["grad"].norm(dim=1, keepdim=True).detach().cpu().numpy()[0, 0],
                cmap="gray",
            )
            grad_img = wandb.Image(fig)
            grad_img = transforms.ToTensor()(grad_img.image)
            plt.close()

            # Log the GT depth image
            fig = plt.figure()
            plt.imshow(
                batch["depth_image"].permute(0, 3, 1, 2)[0, 0].detach().cpu().numpy(),
                cmap="turbo",
            )
            plt.colorbar()
            depth_img = wandb.Image(fig)
            depth_img = transforms.ToTensor()(depth_img.image)
            plt.close()

            image_dict.update(
                {
                    "Grad": grad_img,
                    "Target Depth": depth_img,
                }
            )

            model_outputs["depth_normalized"] = model_outputs["depth"].clone()

            for key in ["depth", "rgb_loss_image", "depth_pred", "depth_normalized"]:
                if key in model_outputs and key not in ["directions_norm", "eik_grad"]:

                    fig = plt.figure()

                    if key != "depth" and key != "depth_pred":
                        plt.imshow(
                            model_outputs[key][0, :, :, :].detach().cpu().numpy(),
                            cmap="turbo",
                        )
                    else:
                        plt.imshow(
                            model_outputs[key][0, :, :, 0].detach().cpu().numpy(),
                            vmin=batch["depth_image"].min(),
                            vmax=batch["depth_image"].max(),
                            cmap="turbo",
                        )
                    plt.colorbar()

                    img = wandb.Image(fig)
                    img = transforms.ToTensor()(img.image)

                    image_dict[key] = img
                    plt.close()

                elif key in batch:

                    fig = plt.figure()

                    if key != "depth":
                        plt.imshow(batch[key][0, :, :, :].detach().cpu().numpy())
                    else:
                        plt.imshow(
                            batch[key][0, :, :, 0].detach().cpu().numpy(),
                            vmin=batch["depth_image"].min(),
                            vmax=batch["depth_image"].max(),
                            cmap="turbo",
                        )
                    plt.colorbar()

                    img = wandb.Image(fig)
                    img = transforms.ToTensor()(img.image)

                    image_dict[key] = img
                    plt.close()
                elif key in misc:

                    fig = plt.figure()

                    if key != "depth" and key != "depth_pred":
                        plt.imshow(misc[key][0, :, :, :].detach().cpu().numpy())
                    else:
                        plt.imshow(
                            misc[key][0, :, :, 0].detach().cpu().numpy(),
                            vmin=batch["depth_image"].min(),
                            vmax=batch["depth_image"].max(),
                            cmap="turbo",
                        )
                    plt.colorbar()

                    img = wandb.Image(fig)
                    img = transforms.ToTensor()(img.image)

                    image_dict[key] = img
                    plt.close()

        # Convert to H, W, C
        for key, value in image_dict.items():
            image_dict[key] = value.permute(1, 2, 0)

        # if not get_image_dict:
        #     return model_outputs, loss_dict, metrics_dict, image_dict, gaussian_stats

        combined_mask = misc["mask_rgb"]

        # Move all tensors to GPU
        batch["image"] = batch["image"].to(self.device)
        model_outputs["rgb"] = model_outputs["rgb"].to(self.device)

        # Mask out the GT image ebfore computing the metrics
        batch["image"] = batch["image"] * combined_mask
        model_outputs["rgb"] = model_outputs["rgb"] * combined_mask

        # Compute metrics
        metrics_dict.update(self.model.get_metrics_dict(model_outputs, batch))

        # Add time metrics
        metrics_dict.update(
            {
                "datafetch_time": datafetch_time,
                "render_time": render_time,
                "loss_time": loss_time,
            }
        )

        if "timesteps" in misc:
            metrics_dict.update(
                {
                    "timesteps": (
                        misc["timesteps"].squeeze()[0]
                        if misc["timesteps"].squeeze().numel() > 1
                        else misc["timesteps"].squeeze()
                    ),
                    "grad_norm": misc["grad"][0].norm(),
                    "weight": (misc["w"].squeeze()[0] if misc["w"].squeeze().numel() > 1 else misc["w"].squeeze()),
                }
            )

        if self.config.input_view_constraint:
            input_view_batch = self.datamanager.get_i_train(0)

            outputs_input_view = self.model(input_view_batch["camera"])

            input_view_batch = input_view_batch["batch"]

            # Compute the RGB loss for the input view
            input_gt_bchw = (
                input_view_batch["image"].permute(0, 3, 1, 2).to(outputs_input_view["rgb"].device)
            )  # b h w c -> b c h w
            input_depth_gt_bchw = (
                input_view_batch["depth_image"].permute(0, 3, 1, 2).to(outputs_input_view["rgb"].device)
            )  # b h w c -> b c h w

            # print(input_gt_bchw.shape, outputs_input_view['rgb'].shape)
            rgb_loss = torch.nn.functional.mse_loss(outputs_input_view["rgb"], input_gt_bchw)
            perceptual_loss = self.lpips_loss(outputs_input_view["rgb"] * 2 - 1, input_gt_bchw * 2 - 1)
            ssim_loss_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(
                outputs_input_view["rgb"].device
            )

            # RGB anchor loss
            loss_dict["rgb_loss_input_view"] = self.model.config.lambda_input_constraint_l2 * rgb_loss
            # loss_dict['rgb_loss_input_view'] = self.model.config.lambda_input_constraint_perceptual * perceptual_loss
            # loss_dict['rgb_loss_input_view_ssim'] = self.model.config.lambda_input_constraint * (1 - ssim_loss_fn(outputs_input_view['rgb'], input_gt_bchw))

            # Depth anchor loss
            depth_bchw = outputs_input_view["depth"]

            if self.config.input_view_depth_constraint:

                depth_pred_model = self.model.depth_guidance.pred_one_step(rgb_in=input_gt_bchw)

                if self.config.input_view_depth_constraint_type == "pearson":
                    # pearson pss
                    depth_input_loss = depth_pearson_loss(src=depth_bchw, tgt=depth_pred_model.detach())
                elif self.config.input_view_depth_constraint_type == "ranking":
                    # ranking loss
                    depth_input_loss = depth_ranking_loss(
                        rendered_depth=depth_bchw,
                        sampled_depth=depth_pred_model.detach(),
                    )
                elif self.config.input_view_depth_constraint_type == "ranking_multi_patch":
                    # ranking loss - multiple patches
                    depth_input_loss = depth_ranking_loss_multi_patch(
                        rendered_depth=depth_bchw,
                        sampled_depth=depth_pred_model.detach(),
                        num_patches=16,
                        num_pairs=1024,
                    )
                else:
                    raise ValueError("Invalid depth constraint type")

                loss_dict["depth_loss_input_view"] = self.model.config.lambda_input_constraint_depth * depth_input_loss

                # Log the depth
                if get_image_dict:
                    fig = plt.figure()

                    plt.imshow(
                        depth_bchw[0, 0].detach().cpu().numpy(),
                        cmap="turbo",
                        vmin=input_depth_gt_bchw.min(),
                        vmax=input_depth_gt_bchw.max(),
                    )
                    plt.colorbar()

                    img = wandb.Image(fig)
                    img = transforms.ToTensor()(img.image)

                    image_dict["Depth Input View"] = img.permute(1, 2, 0)

                    plt.close()

                    fig = plt.figure()

                    plt.imshow(depth_pred_model[0, 0].detach().cpu().numpy(), cmap="turbo")
                    plt.colorbar()

                    img = wandb.Image(fig)
                    img = transforms.ToTensor()(img.image)

                    image_dict["Depth Input View Pred"] = img.permute(1, 2, 0)

                    plt.close()

            # Add input view renders to image dict
            image_dict.update(
                {
                    "Input View": outputs_input_view["rgb"][0].float().permute(1, 2, 0),  # H W C
                    "Input View GT": input_gt_bchw[0].float().permute(1, 2, 0),  # H W C
                }
            )

        return model_outputs, loss_dict, metrics_dict, image_dict, gaussian_stats

    def post_optimizer_step(self, step, model_outputs, optimizers):

        # densify and prune
        if step >= self.config.density_start_iter and step <= self.config.density_end_iter:

            viewspace_point_tensors, visibility_filters, radii = (
                model_outputs["viewspace_points"],
                model_outputs["visibility_filter"],
                model_outputs["radii"],
            )

            visibility_filter_combined = visibility_filters.any(dim=0)
            radii = radii.max(dim=0).values

            # self.model.gaussian_model.max_radii2D[visibility_filter_combined] = torch.max(self.model.gaussian_model.max_radii2D[visibility_filter_combined], radii[visibility_filter_combined])
            self.model.gaussian_model.add_densification_stats(viewspace_point_tensors, visibility_filters)

            densification_counts = {}
            if step % self.config.densification_interval == 0:

                # pdb.set_trace()

                # size_threshold = 20 if self.step > self.config.opacity_reset_interval else None
                densification_counts = self.model.gaussian_model.densify(
                    self.config.densify_grad_threshold,
                    min_opacity=0.1,
                    extent=0.5,
                    max_screen_size=1,
                    optimizers=optimizers.optimizers,
                )

                if self.config.enable_prune:
                    prune_count = self.model.gaussian_model.prune(
                        min_opacity=0.5,
                        extent=0.5,
                        max_screen_size=1,
                        optimizers=optimizers.optimizers,
                    )
                    densification_counts.update(prune_count)

            return densification_counts

        return {}

    def get_eval_loss_dict(self, step: int):
        return {}

    @torch.no_grad()
    def inpaint_all_holes(self):

        raise NotImplementedError("not verified")

        def setup_inpainting_model():
            from diffusers import StableDiffusionInpaintPipeline

            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            )
            pipe.to("cuda")

            return pipe

        def setup_depth_model():
            depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=True).to("cuda")
            depth_model.eval()

            return depth_model

        # Go through the entire dataset - retrieving only camera poses
        num_cameras = len(self.datamanager.train_dataset)

        # Setup models
        inpainting_model = setup_inpainting_model()
        depth_model = setup_depth_model()

        # Go through the entire dataset
        masks = []
        for i in tqdm(range(num_cameras)):
            batch = self.datamanager.get_i_train(i)
            model_outputs = self.model(batch["camera"])
            mask = (model_outputs["depth"] == 0).float()[0]
            masks.append(mask)

        # Sort by mask size
        sorted_idxs = np.argsort([torch.sum(mask) for mask in masks])[::-1]

        # For each camera pose, render the image and depth
        for i in tqdm(range(10)):

            i = sorted_idxs[i]
            batch = self.datamanager.get_i_train(i)

            # Render the image and depth
            model_outputs = self.model(batch["camera"])
            rgb = model_outputs["render"]
            depth = model_outputs["depth"]
            mask = (depth == 0).float()
            pose = batch["c2w"]

            # Compute mask
            mask = model_outputs["depth"] < 0

            # Inpaint the image
            inpainted_image = inpainting_model(
                image=rgb * 2 - 1,
                mask_image=mask.float(),
                prompt=self.config.prompt,
                negative_prompt="ugly, blurry, text, low resolution, dark",
                num_inference_steps=25,
                output_type="pt",
            ).images
            inpainted_image = (inpainted_image.float() + 1) / 2

            # Estimate the depth
            disp_pred = depth_model.forward(inpainted_image)

            # Align the depth
            aligned_depth = align_depth(disp_pred, model_outputs["depth"], 1 - mask, fuse=False)
            aligned_depth = bilateral_filter(inpainted_image, aligned_depth.unsqueeze(0), depth_threshold=0.01)[0]

            # Update hte point cloud
            self.model.gaussian_model.backproject_and_combine(
                inpainted_image[0].permute(1, 2, 0), aligned_depth.float(), mask, pose
            )

            # Go through the entire dataset
            masks = []
            for i in tqdm(range(num_cameras)):
                batch = self.datamanager.get_i_train(i)
                model_outputs = self.model(batch["camera"])
                mask = (model_outputs["depth"] == 0).float()[0]
                masks.append(mask)

        del inpainting_model
        del depth_model

        # Cleanup CUDA cache
        torch.cuda.empty_cache()

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }

        # self.model.update_to_step(step)
        self.load_state_dict(state, strict=False)

    # Reset the optimizer
    def reset_params(self, optimizers):

        # self.model.gaussian_model.reload_optimizer(optimizers)
        params = self.model.get_param_groups()

        for key in optimizers.keys():

            if key in ["guidance"]:
                continue

            self.model.gaussian_model.replace_tensor_to_optimizer(params[key][0], key, optimizers)


def apply_depth_smoothing(image, mask):

    def dilate(x, k=3):
        x = torch.nn.functional.conv2d(
            x.float()[None, None, ...],
            torch.ones(1, 1, k, k).to(x.device),
            padding="same",
        )
        return x.squeeze() > 0

    def sobel(x):
        flipped_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(x.device)
        flipped_sobel_x = torch.stack([flipped_sobel_x, flipped_sobel_x.t()]).unsqueeze(1)

        x_pad = torch.nn.functional.pad(x.float()[None, None, ...], (1, 1, 1, 1), mode="replicate")

        x = torch.nn.functional.conv2d(x_pad, flipped_sobel_x, padding="valid")
        dx, dy = x.unbind(dim=-3)
        return torch.sqrt(dx**2 + dy**2).squeeze()
        # new content is created mostly in x direction, sharp edges in y direction are wanted (e.g. table --> wall)
        # return dx.squeeze()

    edges = sobel(mask)
    dilated_edges = dilate(edges, k=21)

    img_numpy = image.float().cpu().numpy()
    blur_bilateral = cv2.bilateralFilter(img_numpy, 5, 140, 140)
    blur_gaussian = cv2.GaussianBlur(blur_bilateral, (5, 5), 0)
    blur_gaussian = torch.from_numpy(blur_gaussian).to(image)

    image_smooth = torch.where(dilated_edges, blur_gaussian, image)
    return image_smooth


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def align_depth(disp, depth_gt, mask, fuse=True):

    # mask - 1 = not inpaitned (not holes), 0 = inpainted (holes)

    if type(depth_gt) != torch.Tensor:
        depth_gt = torch.from_numpy(depth_gt).to(disp).unsqueeze(0)

    if type(mask) != torch.Tensor:
        mask = torch.from_numpy(mask).to(disp).unsqueeze(0)

    assert depth_gt.shape == mask.shape
    assert depth_gt.shape == disp.shape
    assert mask.max() <= 1

    # Convert depth gt to disparity
    disp_gt = 1 / depth_gt
    disp = disp * torch.median(disp_gt[mask > 0]) / torch.median(disp[mask > 0])
    depth = 1 / disp

    # # Compute the scale and shift factor to align the disparity
    # assert torch.isnan(disp).any() == False, "Disp is NaN"
    # assert torch.isnan(disp_gt).any() == False, "Disp GT is NaN"

    # print(disp.shape, disp_gt.shape, mask.shape)
    scale, shift = compute_scale_and_shift(depth, depth_gt, depth_gt > 0)
    depth = scale[:, None, None] * depth + shift[:, None, None]

    # assert torch.isnan(scale).any() == False, "Scale is NaN"
    # assert torch.isnan(shift).any() == False, "Shift is NaN"

    # # Apply the scale and shift factor to the disparity
    # disp = scale[:, None, None] * disp + shift[:, None, None]

    # Convert the disparity back to depth

    if fuse:
        # Fuse the depth and depth gt
        depth = depth_gt * mask + depth * (1 - mask)

        depth = depth[0]
        depth_gt = depth_gt[0]
        mask = mask[0]
        for i in range(500):
            depth = apply_depth_smoothing(depth.float(), mask)
            depth = depth_gt * mask + depth * (1 - mask)

    depth = depth[0]

    return depth


def bilateral_filter(img, depth, depth_threshold=0.01):

    B = img.shape[0]
    output = torch.zeros_like(depth)

    # Config for 3D photography
    config = {}
    config["gpu_ids"] = 0
    config["extrapolation_thickness"] = 60
    config["extrapolate_border"] = True
    config["depth_threshold"] = depth_threshold
    config["redundant_number"] = 12
    config["ext_edge_threshold"] = 0.002
    config["background_thickness"] = 70
    config["context_thickness"] = 140
    config["background_thickness_2"] = 70
    config["context_thickness_2"] = 70
    config["log_depth"] = True
    config["depth_edge_dilate"] = 10
    config["depth_edge_dilate_2"] = 5
    config["largest_size"] = 512
    config["repeat_inpaint_edge"] = True
    config["ply_fmt"] = "bin"

    config["save_ply"] = False
    config["save_obj"] = False

    config["sparse_iter"] = 10
    config["filter_size"] = [7, 7, 5, 5, 5, 5, 5, 5, 5, 5]
    config["sigma_s"] = 4.0
    config["sigma_r"] = 0.5

    for i in range(B):

        # Convert to numpy
        img_np = img[i].detach().cpu().permute(1, 2, 0).numpy()
        depth_np = depth[i].detach().cpu().numpy()

        # Run the filter
        vis_images, vis_depths = sparse_bilateral_filtering(
            depth_np,
            img_np * 255,
            config,
            num_iter=config["sparse_iter"],
            spdb=False,
            HR=True,
        )

        # Convert back to torch
        vis_depths = torch.from_numpy(vis_depths[-1]).float().to(depth.device)

        # Plot the output
        # plt.imshow(vis_images[-1])
        # plt.title("vis_image")
        # plt.colorbar()
        # plt.show()

        # Save the output
        output[i] = vis_depths

    return output
