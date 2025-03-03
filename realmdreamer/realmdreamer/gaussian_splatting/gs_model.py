from __future__ import annotations

import json
import math
import os
import pdb
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Tuple, Type, Union

import kornia
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)
from jaxtyping import Bool, Float
from kornia.color.colormap import AUTUMN
from kornia.enhance import sharpness
from kornia.filters import gaussian_blur2d, laplacian
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import (
    DepthLossType, L1Loss, MSELoss, ScaleAndShiftInvariantLoss, depth_loss,
    distortion_loss, interlevel_loss, lossfun_outer, orientation_loss,
    pred_normal_loss, ray_samples_to_sdist,
    scale_gradients_by_distance_squared, tv_loss)
from nerfstudio.model_components.ray_samplers import (ProposalNetworkSampler,
                                                      UniformSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   NormalsRenderer,
                                                   RGBRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import ViewerSlider
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.image import (MultiScaleStructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.depth import (depth_pearson_loss, depth_ranking_loss,
                         depth_ranking_loss_multi_patch_masked,
                         patch_pearson_loss)

from realmdreamer.guidance.geowizard_guidance import GeoWizardGuidance
from realmdreamer.guidance.marigold_guidance import (MarigoldGuidance,
                                                     align_depths)
from realmdreamer.guidance.sd_guidance import StableDiffusionGuidance
from realmdreamer.guidance.sd_inpainting_guidance import \
    StableDiffusionInpaintingGuidance

from .gs_camera import Camera as GaussianSplattingCamera
from .gs_camera_utils import ns2gs_camera
from .gs_field import GaussianSplattingField, GaussianSplattingFieldConfig
from .gs_graphics_utils import focal2fov, fov2focal, getWorld2View2
from .gs_sh_utils import eval_sh
from .gs_utils import wide_sigmoid

# from realmdreamer.guidance.sd_inpainting_guidance_ism import \
# StableDiffusionInpaintingISMGuidance




@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: GaussianSplatting)

    gaussian_model: GaussianSplattingFieldConfig = GaussianSplattingFieldConfig()
    """Config for the Gaussian model."""

    background_color: str = "white"
    """Background color for the rendered images. Either "black" or "white"."""

    sh_degree: int = 0
    """Maximum SH degree to use for rendering."""

    pcd_path: str = None
    """Path to the model to load."""

    full_precision: bool = False
    """Whether to use full precision for SDS guidance."""

    guidance: str = "sds_inpainting"
    """sds or sds_inpainting"""

    guidance_scale: float = 100.0
    """Guidance scale for SDS guidance."""

    img_guidance_scale: float = 1.0
    """Guidance scale for image conditioning in Inpainting NFSD/VSD"""

    min_step_percent: float = 0.02
    """Minimum step percent for SDS guidance."""

    max_step_percent: float = 0.98
    """Maximum step percent for SDS guidance."""

    ignore_mask: bool = False
    """Whether or not to ignore the inpainting mask"""

    anneal: bool = False
    """Whether or not to anneal timesteps"""

    prolific_anneal: bool = False
    """Whether or not to anneal timesteps according to prolific dreamer"""

    loss_type: str = "noise"
    """Either noise, one_step, or multi_step, or nfsd - determines if regular SDS, or equivalent denoised SDS or multi-step SDS (Ref Sparsefusion) - Only supports sd_inpainting"""

    invert_ddim: bool = False
    """Whether to use DDIM inversion for computing noisy latents"""

    invert_after_step: bool = False
    """Use DDIM inversion only after a particular step"""

    invert_step_ratio: float = 0.1
    """Step to start using DDIM inversion"""

    ddim_invert_method: str = "ddim" or "pseudo_ddim" or "ddim_0"
    """Whether to use DDIM inversion for computing noisy latents or DDPM or pseudo ddim"""

    num_steps_sample: int = 10
    """Number of steps to sample for multi-step SDS"""

    fixed_num_steps: bool = False
    """Whether to use a fixed number of steps for multi-step SDS"""

    set_sds_weight_l2_and_perceptual: bool = False
    """Whether to set the weight of the SDS loss (from the alpha) for the l2 and perceptual loss"""

    lambda_sds: float = 0.1
    """Multiplier for SDS loss."""

    lambda_rgb: float = 1000.0
    """Multiplier for RGB loss."""

    lambda_depth: float = 0.0
    """Multiplier for depth loss."""

    lambda_opaque: float = 0.0
    """Multiplier for opaqueness loss."""

    lambda_one_step: float = 0.0
    """Multiplier for RGB loss (l2) with one step predictions"""

    lambda_one_step_l1: float = 0.0
    """Multiplier for L1 loss with one step predictions"""

    lambda_one_step_perceptual: float = 0.0
    """Multiplier for perceptual loss with one step predictions"""

    lambda_one_step_ssim: float = 0.0
    """Multiplier for SSIM loss with one step predictions"""

    lambda_input_constraint_l2: float = 0.0
    """Multiplier for the anchor loss on the input view"""

    lambda_input_constraint_perceptual: float = 0.0
    """Multiplier for the perceptual loss on the input view"""

    lambda_input_constraint_depth: float = 0.0
    """Multiplier for the depth loss on the input view"""

    use_sigmoid: bool = False
    """Whether to use sigmoid on color activations"""

    average_colors: bool = False
    """Average color of that are near each other before rendering"""

    depth_guidance: bool = False
    """Whether to use depth guidance for SDS"""

    depth_guidance_multi_step: bool = True
    """Whether to use multi-step depth guidance for SDS"""

    load_depth_guidance: bool = False
    """Whether to load depth guidance model for the diffusion model - can be used if even if depth_guidance is False"""

    lambda_depth_sds: float = 100.0
    """The relative weight to the first SDS to use for depth SDS"""

    depth_loss: Literal["pearson", "patch_pearson", "ranking", "ranking_multi_patch"] = "pearson"
    """If using depth loss, whether to use MSE (l2), ranking, or pearson loss or patch pearson loss"""

    depth_patch_size: int = 64
    """Patch size for patch pearson/ranking loss"""

    depth_num_patches: int = 64
    """Number of patches to use for ranking loss multi patch"""

    depth_patch_percent: float = 0.1
    """Percentage of patches to use for patch pearson loss"""

    depth_num_pairs: int = 1024
    """Number of pairs to use for ranking loss"""

    load_dreambooth: bool = False
    """Whether to load dreambooth model for the diffusion model"""

    sharpen_in_post: bool = False
    """Whether to sharpen the diffusion model predictions as post processing"""

    sharpen_in_post_factor: float = 1.0
    """Factor to sharpen the diffusion model predictions as post processing"""

    # IGNORE:

    inference_only: bool = False
    """Whether to only use the model for inference"""

    occluded_rand_init: bool = True
    """Whether to occlude the point cloud for random initialization"""


class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianSplatting(Model):
    config: GaussianSplattingModelConfig
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        device,
        load_iteration: int = -1,
        orientation_transform: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(config, scene_box, num_train_data)

        self.config = config

        if not self.config.inference_only:
            self.guidance.to(self.device)

        self.gaussian_model.to(self.device)
        self.gaussian_model.xyz_gradient_accum.to(self.device)
        self.gaussian_model.denom.to(self.device)

        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        self.opacity_modifier = ViewerSlider(name="Opacity Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.scale_modifier = ViewerSlider(name="Scale Slider", default_value=1.0, min_value=0.0, max_value=1.0)

    def setup_diffusion(self, dreambooth_path=None):

        if self.config.guidance == "sds":
            self.guidance = StableDiffusionGuidance(
                device="cuda",
                full_precision=self.config.full_precision,
                model_path=dreambooth_path,
                min_step_percent=self.config.min_step_percent,
                max_step_percent=self.config.max_step_percent,
                guidance_scale=self.config.guidance_scale,
                anneal=self.config.anneal,
                prolific_anneal=self.config.prolific_anneal,
                invert_ddim=self.config.invert_ddim,
                num_steps_sample=self.config.num_steps_sample,
                ddim_invert_method=self.config.ddim_invert_method,
                fixed_num_steps=self.config.fixed_num_steps,
            )
        elif self.config.guidance == "sds_inpainting":
            self.guidance = StableDiffusionInpaintingGuidance(
                device="cuda",
                full_precision=self.config.full_precision,
                min_step_percent=self.config.min_step_percent,
                max_step_percent=self.config.max_step_percent,
                guidance_scale=self.config.guidance_scale,
                img_guidance_scale=self.config.img_guidance_scale,
                anneal=self.config.anneal,
                prolific_anneal=self.config.prolific_anneal,
                invert_ddim=self.config.invert_ddim,
                num_steps_sample=self.config.num_steps_sample,
            )
        else:
            raise NotImplementedError(f"{self.config.guidance} is not a valid guidance model")

        if self.config.depth_guidance or self.config.load_depth_guidance:
            # self.depth_guidance = MarigoldGuidance(device='cuda')
            self.depth_guidance = GeoWizardGuidance(device="cuda")

        # If anneal is set to none, disable time prior
        if not self.config.anneal and self.config.guidance != "dummy":
            self.guidance.time_prior = None

    def populate_modules(self):
        super().populate_modules()

        # Get scene name from pcd
        scene_name = os.path.basename(os.path.dirname(self.config.pcd_path))

        # load gaussian model
        self.gaussian_model = self.config.gaussian_model.setup()

        self.gaussian_model.load_pcd(
            os.path.join(self.config.pcd_path),
            occluded_rand_init=self.config.occluded_rand_init,
            device="cuda",
            use_sigmoid=self.config.use_sigmoid,
        )
        # self.gaussian_model.init_random_from_pcd(os.path.join(self.config.pcd_path), occluded_rand_init=False)

        self.gaussian_model.xyz_gradient_accum = torch.zeros((self.gaussian_model.get_xyz.shape[0], 1), device="cuda")
        self.gaussian_model.denom = torch.zeros((self.gaussian_model.get_xyz.shape[0], 1), device="cuda")

        # Set up losses
        # self.rgb_loss = MSELoss()
        self.rgb_loss = L1Loss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        self.lpips = lpips.LPIPS(net="vgg").to("cuda")

        # Diffusion Guidance
        self.diffusion_setup = True

        if not self.config.load_dreambooth:
            self.setup_diffusion()
        else:

            dreambooth_path = self.config.pcd_path.replace("pointcloud.ply", "dreambooth")
            assert os.path.exists(dreambooth_path), f"Dreambooth path {dreambooth_path} does not exist"

            self.setup_diffusion(dreambooth_path)

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:

        viewpoint_camera = ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
        )

        render = render_results["render"]
        depth = render_results["depth"]
        alpha = render_results["alpha"]

        rgb = torch.permute(torch.clamp(render, max=1.0), (1, 2, 0))
        depth = torch.permute(depth, (1, 2, 0))
        return {
            "rgb": rgb,
            "depth": depth,
        }

    def render(
        self,
        viewpoint_camera,
        pc,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussian_model.get_xyz,
                dtype=self.gaussian_model.get_xyz.dtype,
                requires_grad=True,
                device=pc.get_xyz.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(self.device)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        # print(pc.get_xyz.device, pc.get_features.device, viewpoint_camera.camera_center.device, self.device)

        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        if self.config.average_colors:
            shs_view = (
                pc.get_averaged_features(num_neighbours=2).transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            )

        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        if self.config.use_sigmoid:
            colors_precomp = sh2rgb
            colors_precomp = wide_sigmoid(colors_precomp)

        # print(means3D.device, means3D.device, colors_precomp.device, opacity.device, scales.device, rotations.device)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        visibility_filter = radii > 0

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["xyz"] = [self.gaussian_model._xyz]
        param_groups["f_dc"] = [self.gaussian_model._features_dc]
        param_groups["f_rest"] = [self.gaussian_model._features_rest]
        param_groups["opacity"] = [self.gaussian_model._opacity]
        param_groups["scaling"] = [self.gaussian_model._scaling]
        param_groups["rotation"] = [self.gaussian_model._rotation]

        if self.config.guidance == "vsd":
            print("Adding guidance parameters to param_groups")
            param_groups["guidance"] = list(self.guidance.parameters())

        return param_groups

    def named_parameters(self, recurse=False):
        params = {}

        params["xyz"] = self.gaussian_model._xyz
        params["f_dc"] = self.gaussian_model._features_dc
        params["f_rest"] = self.gaussian_model._features_rest
        params["opacity"] = self.gaussian_model._opacity
        params["scaling"] = self.gaussian_model._scaling
        params["rotation"] = self.gaussian_model._rotation

        if self.config.guidance == "vsd":
            params["guidance"] = list(self.guidance.parameters())

        # Convert dict to iterator
        params = params.items()

        return params

    def forward_single(self, viewpoint_camera) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Run forward for a single camera pose
        """

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=self.device)

        # random background
        random_background = torch.tensor(np.random.uniform(0, 1, (3,)), dtype=torch.float32, device=self.device)

        start = time.time()
        out = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            # bg_color=background
            bg_color=random_background,
        )
        end = time.time()

        # print(out['render'].shape, end - start, "IMAGE")

        image = out["render"]
        depth = out["depth"]
        alpha = out["alpha"]
        viewspace_points = out["viewspace_points"]
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]

        return {
            "rgb": image,
            "depth": depth,
            "alpha": alpha,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    @torch.cuda.amp.autocast(False)
    def forward(self, cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a camera pose
        Args:
            - c2w: camera poses
        """

        viewpoint_cameras = ns2gs_camera(cameras, device=self.device)

        # assert not isinstance(viewpoint_cameras, list), "Cameras are a list is a list"
        if isinstance(viewpoint_cameras, list):

            outputs = {
                "rgb": [],
                "depth": [],
                "alpha": [],
                "viewspace_points": [],
                "visibility_filter": [],
                "radii": [],
            }

            for viewpoint_camera in viewpoint_cameras:
                out = self.forward_single(viewpoint_camera)
                for key in out.keys():
                    outputs[key].append(out[key])

            for key in outputs.keys():
                if key != "viewspace_points":
                    outputs[key] = torch.stack(outputs[key], dim=0)
        else:

            outputs = self.forward_single(viewpoint_cameras)

            for key in outputs.keys():
                if key == "viewspace_points":
                    outputs[key] = [outputs[key]]
                else:
                    outputs[key] = outputs[key].unsqueeze(0)

        return outputs

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}

        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        return metrics_dict

    def masked_rgb_loss(self, image, gt, depth, mask):
        """Compute RGB loss on masked image."""

        """
            Sometimes the rgb image will have holes in it, due to holes in point cloud and missing depth values - mask these out as well
        """

        # Depth - B, H, W, 1
        mask_depth = (depth > 0) & ~mask

        if mask_depth.shape[1] != image.shape[1] or mask_depth.shape[2] != image.shape[2]:
            # Erode the mask_depth before resizing
            mask_depth = kornia.morphology.erosion(
                mask_depth.permute(0, 3, 1, 2).float(),
                torch.ones((5, 5)).to(mask_depth.device),
            )  # B C H W
            mask_depth = kornia.morphology.erosion(mask_depth, torch.ones((5, 5)).to(mask_depth.device))

            mask_depth = F.interpolate(
                mask_depth.float(),
                size=(image.shape[1], image.shape[2]),
                mode="nearest",
            )
            mask_depth = mask_depth.bool().permute(0, 2, 3, 1)  # B H W C

        return self.rgb_loss(image * mask_depth, gt * mask_depth), mask_depth

    def masked_depth_loss(self, depth, gt, mask):
        """Compute depth loss on masked depth."""

        mask_depth = (depth > 0) & ~mask

        depth_masked = depth[mask_depth]
        gt_masked = gt[mask_depth]

        return F.mse_loss(depth_masked, gt_masked, reduction="sum") / depth_masked.numel()

    def get_loss_dict(self, outputs, batch, prompt, c2w, step_ratio):

        if not self.diffusion_setup:
            self.setup_diffusion()
            self.diffusion_setup = True

        loss_dict = {}
        misc = {}

        batch_size = batch["inpainting_mask"].shape[0]

        # Get image from point cloud and inpainting mask
        image = batch["image"].to(self.device)  # B H W C

        batch["inpainting_mask"] = batch["inpainting_mask"].to(self.device)
        batch["inpainting_mask_2"] = batch["inpainting_mask_2"].to(self.device)
        batch["depth_image"] = batch["depth_image"].float().to(self.device)

        # Mask override
        if self.config.ignore_mask:
            batch["inpainting_mask"] = torch.ones_like(batch["inpainting_mask"]) > 0
            batch["inpainting_mask_2"] = torch.ones_like(batch["inpainting_mask_2"]) > 0

        # inverted mask is inverse of the combined masks - 1 = RGB, 0 = hole in point cloud
        inverted_inpainting_mask = ~batch["inpainting_mask"] & ~batch["inpainting_mask_2"]

        rendered_rgb_bhwc = outputs["rgb"].permute(0, 2, 3, 1)
        rendered_rgb_bchw = outputs["rgb"]
        rendered_depth_bhwc = outputs["depth"].permute(0, 2, 3, 1)
        rendered_depth_bchw = outputs["depth"]

        for key in ["rgb", "depth"]:

            outputs[key] = outputs[key].permute(0, 2, 3, 1)

        # Only compute SDS loss during training
        if self.training:

            if self.config.invert_ddim:

                if self.config.invert_after_step and step_ratio < self.config.invert_step_ratio:
                    self.guidance.cfg.invert = False
                else:
                    self.guidance.cfg.invert = True

            sds_start_time = time.time()

            # Pass the RGB image through the diffusion model
            if self.config.guidance == "sds":

                fn = self.guidance.multi_step

                loss_dict, misc = fn(
                    rgb=rendered_rgb_bhwc,
                    prompt=prompt,
                    mask=batch["inpainting_mask"],
                    rgb_as_latents=False,
                    current_step_ratio=step_ratio,
                )

            elif self.config.guidance == "sds_inpainting":

                fn = self.guidance.multi_step

                loss_dict, misc = fn(
                    rgb=rendered_rgb_bhwc,
                    og_rgb=rendered_rgb_bhwc,
                    prompt=prompt,
                    mask=batch["inpainting_mask"],
                    rgb_as_latents=False,
                    current_step_ratio=step_ratio,
                    mask_grad=True,
                )

            end_sds_time = time.time()

            ssim_loss_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(rendered_rgb_bchw.device)

            # SDS loss
            loss_dict["loss_sds"] = self.config.lambda_sds * loss_dict["loss_sds"]

            # SDS - RGB loss - either multi step or one step should be there
            if self.config.sharpen_in_post:

                key = "multi_step_pred" if "multi_step_pred" in misc.keys() else "one_step_pred"
                misc[key] = sharpness(misc[key], factor=self.config.sharpen_in_post_factor)

            if "multi_step_pred" in misc.keys():

                weight = misc["w"].squeeze()[0] if misc["w"].squeeze().numel() > 1 else misc["w"].squeeze()

                if not self.config.set_sds_weight_l2_and_perceptual:
                    weight = 1.0

                loss_dict["loss_multi_step"] = (
                    self.config.lambda_one_step
                    * weight
                    * 0.5
                    * F.mse_loss(misc["multi_step_pred"], rendered_rgb_bchw, reduction="sum")
                    / batch_size
                )
                loss_dict["loss_multi_step_l1"] = (
                    self.config.lambda_one_step_l1
                    * weight
                    * F.l1_loss(misc["multi_step_pred"], rendered_rgb_bchw, reduction="sum")
                    / batch_size
                )
                loss_dict["loss_multi_step_perceptual"] = (
                    self.config.lambda_one_step_perceptual
                    * weight
                    * self.lpips(
                        misc["multi_step_pred"].detach() * 2 - 1,
                        rendered_rgb_bchw * 2 - 1,
                    ).sum()
                    / batch_size
                )
                loss_dict["loss_multi_step_ssim"] = (
                    self.config.lambda_one_step_ssim
                    * weight
                    * (1 - ssim_loss_fn(misc["multi_step_pred"], rendered_rgb_bchw).sum())
                    / batch_size
                )
            else:

                weight = misc["w"].squeeze()[0] if misc["w"].squeeze().numel() > 1 else misc["w"].squeeze()

                if not self.config.set_sds_weight_l2_and_perceptual:
                    weight = 1.0

                loss_dict["loss_one_step"] = (
                    self.config.lambda_one_step
                    * weight
                    * 0.5
                    * F.mse_loss(misc["one_step_pred"], rendered_rgb_bchw, reduction="sum")
                    / batch_size
                )
                loss_dict["loss_one_step_l1"] = (
                    self.config.lambda_one_step_l1
                    * weight
                    * F.l1_loss(misc["one_step_pred"], rendered_rgb_bchw, reduction="sum")
                    / batch_size
                )
                loss_dict["loss_one_step_perceptual"] = (
                    self.config.lambda_one_step_perceptual
                    * weight
                    * self.lpips(
                        misc["one_step_pred"].detach() * 2 - 1,
                        rendered_rgb_bchw * 2 - 1,
                    ).sum()
                    / batch_size
                )
                loss_dict["loss_one_step_ssim"] = (
                    self.config.lambda_one_step_ssim
                    * weight
                    * (1 - ssim_loss_fn(misc["one_step_pred"], rendered_rgb_bchw).sum())
                    / batch_size
                )

            remaining_loss_s = time.time()

            # RGB loss
            loss_dict["loss_rgb"], mask_rgb_anchor = self.masked_rgb_loss(
                rendered_rgb_bhwc,
                image,
                batch["depth_image"].to(self.device),
                batch["inpainting_mask"],
            )
            loss_dict["loss_rgb"] = self.config.lambda_rgb * loss_dict["loss_rgb"]
            misc["mask_rgb"] = mask_rgb_anchor

            # Depth Loss
            # print(outputs['depth'].shape, batch['depth_image'].shape)
            if (
                rendered_depth_bhwc.shape[1] != batch["depth_image"].shape[1]
                or rendered_depth_bhwc.shape[2] != batch["depth_image"].shape[2]
            ):
                outputs["depth"] = F.interpolate(
                    outputs["depth"],
                    size=(batch["depth_image"].shape[1], batch["depth_image"].shape[2]),
                    mode="nearest",
                )
                rendered_depth_bhwc = outputs["depth"].permute(0, 2, 3, 1)  # B C H W -> B H W C

            if (~batch["inpainting_mask"]).sum() > 0:  # If the mask is all 1s, don't compute the depth loss
                loss_dict["loss_depth"] = self.config.lambda_depth * self.masked_depth_loss(
                    rendered_depth_bhwc,
                    batch["depth_image"].to(self.device),
                    batch["inpainting_mask"],
                )

            # Opaqueness Loss
            clamped_opacity = torch.clamp(self.gaussian_model.get_opacity, min=1e-5, max=1.0 - 1e-5)
            loss_dict["loss_opaque"] = self.config.lambda_opaque * F.binary_cross_entropy(
                clamped_opacity, clamped_opacity
            )

            # Depth guidance loss
            if self.config.depth_guidance:

                fn_depth = (
                    self.depth_guidance.sample
                    if self.config.depth_guidance_multi_step
                    else self.depth_guidance.pred_one_step
                )

                # depth_pred_normalized = self.depth_guidance.pred_one_step(outputs['rgb'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                if "multi_step_pred" in misc.keys():
                    depth_pred_normalized = fn_depth(
                        rgb_in=misc["multi_step_pred"],
                        depth_in=outputs["depth"].permute(0, 3, 1, 2),
                        strength=1.0,
                    )
                else:
                    depth_pred_normalized = fn_depth(
                        rgb_in=misc["one_step_pred"],
                        depth_in=outputs["depth"].permute(0, 3, 1, 2),
                        strength=1.0,
                    )

                # mask_align = ~batch['inpainting_mask'] & (batch['depth_image'] > 0)
                mask_align = batch["depth_image"] > 0

                depth_pred_aligned = align_depths(
                    source_depth=depth_pred_normalized,
                    target_depth=outputs["depth"].permute(0, 3, 1, 2),
                    mask=mask_align.permute(0, 3, 1, 2),
                    enforce_scale_positive=True,
                )  # B C H W

                if depth_pred_aligned is not None:

                    misc["depth_pred"] = depth_pred_aligned.permute(0, 2, 3, 1)  # B C H W -> B H W C

                    if self.config.depth_loss == "l2":

                        # Regular L2 Loss
                        loss_dict["loss_depth_sds"] = (
                            self.config.lambda_depth_sds
                            * F.mse_loss(
                                depth_pred_aligned.detach(),
                                outputs["depth"].permute(0, 3, 1, 2),
                                reduction="sum",
                            )
                            / batch_size
                        )

                    elif self.config.depth_loss == "ranking":

                        loss_dict["loss_depth_sds"] = self.config.lambda_depth_sds * depth_ranking_loss(
                            rendered_depth_bchw, depth_pred_normalized.detach()
                        )

                    elif self.config.depth_loss == "ranking_multi_patch":

                        mask_bchw = batch["inpainting_mask"].permute(0, 3, 1, 2)

                        loss_dict["loss_depth_sds"] = depth_ranking_loss_multi_patch_masked(
                            rendered_depth=rendered_depth_bchw,
                            sampled_depth=depth_pred_aligned,
                            mask=mask_bchw,
                            num_patches=self.config.depth_num_patches,
                            num_pairs=self.config.depth_num_pairs,
                            crop_size=(
                                self.config.depth_patch_size,
                                self.config.depth_patch_size,
                            ),
                        )

                    elif self.config.depth_loss == "ranking_smooth":

                        # Ranking Loss
                        margin = 1e-4
                        rendered_depth_bchw = outputs["depth"].permute(0, 3, 1, 2)

                        depth_diff_vertical_render = (
                            rendered_depth_bchw[:, :, 1:, :] - rendered_depth_bchw[:, :, :-1, :]
                        )
                        depth_diff_vertical_gt = (
                            depth_pred_aligned[:, :, 1:, :] - depth_pred_aligned[:, :, :-1, :] + margin
                        )

                        depth_diff_horizontal_render = (
                            rendered_depth_bchw[:, :, :, 1:] - rendered_depth_bchw[:, :, :, :-1]
                        )
                        depth_diff_horizontal_gt = (
                            depth_pred_aligned[:, :, :, 1:] - depth_pred_aligned[:, :, :, :-1] + margin
                        )

                        differing_signs_vertical = torch.sign(depth_diff_vertical_render) != torch.sign(
                            depth_diff_vertical_gt
                        )
                        different_signs_horizontal = torch.sign(depth_diff_horizontal_render) != torch.sign(
                            depth_diff_horizontal_gt
                        )

                        horizontal_ranking_loss = torch.nanmean(
                            depth_diff_horizontal_render[different_signs_horizontal]
                            * torch.sign(depth_diff_horizontal_render[different_signs_horizontal])
                        )

                        vertical_ranking_loss = torch.nanmean(
                            depth_diff_vertical_render[differing_signs_vertical]
                            * torch.sign(depth_diff_vertical_render[differing_signs_vertical])
                        )

                        ranking_loss = (horizontal_ranking_loss + vertical_ranking_loss) / 2

                        loss_dict["loss_depth_sds"] = self.config.lambda_depth_sds * ranking_loss

                    elif self.config.depth_loss == "pearson":

                        rendered_depth = outputs["depth"].reshape(batch_size, -1)
                        pred_depth = depth_pred_aligned.permute(0, 2, 3, 1).reshape(batch_size, -1)

                        pearson_corr_loss = 0
                        for idx in range(batch_size):
                            pearson_corr_loss += 1.0 - pearson_corrcoef(rendered_depth[idx], pred_depth[idx].detach())

                        loss_dict["loss_depth_sds"] = self.config.lambda_depth_sds * pearson_corr_loss

                    elif self.config.depth_loss == "patch_pearson":

                        rendered_depth = rendered_depth_bchw
                        pred_depth = depth_pred_normalized

                        depth_loss = patch_pearson_loss(
                            depth_src=rendered_depth,
                            depth_target=pred_depth.detach(),
                            box_p=self.config.depth_patch_size,
                            p_corr=self.config.depth_patch_percent,
                        )
                        loss_dict["loss_depth_sds"] = self.config.lambda_depth_sds * depth_loss

                    elif self.config.depth_loss == "pearson+ranking":

                        rendered_depth_bchw = outputs["depth"].permute(0, 3, 1, 2)
                        loss_dict["loss_depth_sds"] = self.config.lambda_depth_sds * (
                            depth_ranking_loss(rendered_depth_bchw, depth_pred_aligned)
                            + depth_pearson_loss(rendered_depth_bchw, depth_pred_aligned)
                        )

                else:
                    print("Skipping step...")

            remaining_loss_end = time.time()

        nan_start = time.time()

        # Ensure that there's no NaNs in the loss
        for key in loss_dict.keys():
            loss_dict[key] = torch.nan_to_num(loss_dict[key])

        nan_end = time.time()

        return loss_dict, misc

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Create a new state_dict with only desired submodules
        filtered_state_dict = super().state_dict(destination, prefix, keep_vars)
        for name, param in self.named_parameters():
            if "guidance" not in name:
                filtered_state_dict[prefix + name] = param

        return filtered_state_dict

    def load_state_dict(self, model_dict, strict=True):

        mask = self.gaussian_model.opacity_activation(model_dict["gaussian_model._opacity"]) < 0.95

        # Reinitialize the shape of all the gaussians
        self.gaussian_model.init_random(num_points=model_dict["gaussian_model._xyz"].shape[0])
        super().load_state_dict(model_dict, strict)

    def get_stats(self):

        stats = {}

        stats["num_points"] = self.gaussian_model.get_xyz.shape[0]

        # for each field, log min, max, mean, grad_min, grad_max
        stats["xyz/min"] = torch.min(self.gaussian_model.get_xyz)
        stats["xyz/max"] = torch.max(self.gaussian_model.get_xyz)
        stats["xyz/median"] = torch.median(self.gaussian_model.get_xyz)
        stats["xyz/mean"] = torch.mean(self.gaussian_model.get_xyz)

        stats["xyz/grad_min"] = torch.min(self.gaussian_model.xyz_gradient_accum)
        stats["xyz/grad_max"] = torch.max(self.gaussian_model.xyz_gradient_accum)
        stats["xyz/grad_median"] = torch.median(self.gaussian_model.xyz_gradient_accum)
        stats["xyz/grad_mean"] = torch.mean(self.gaussian_model.xyz_gradient_accum)

        stats["features/min"] = torch.min(self.gaussian_model.get_features)
        stats["features/max"] = torch.max(self.gaussian_model.get_features)
        stats["features/median"] = torch.median(self.gaussian_model.get_features)
        stats["features/mean"] = torch.mean(self.gaussian_model.get_features)

        stats["opacity/min"] = torch.min(self.gaussian_model.get_opacity)
        stats["opacity/max"] = torch.max(self.gaussian_model.get_opacity)
        stats["opacity/median"] = torch.median(self.gaussian_model.get_opacity)
        stats["opacity/mean"] = torch.mean(self.gaussian_model.get_opacity)
        stats["opacity/low_opacity_count"] = torch.sum(self.gaussian_model.get_opacity < 0.1)

        stats["scaling/min"] = torch.min(self.gaussian_model.get_scaling)
        stats["scaling/max"] = torch.max(self.gaussian_model.get_scaling)
        stats["scaling/median"] = torch.median(self.gaussian_model.get_scaling)
        stats["scaling/mean"] = torch.mean(self.gaussian_model.get_scaling)

        stats["rotation/min"] = torch.min(self.gaussian_model.get_rotation)
        stats["rotation/max"] = torch.max(self.gaussian_model.get_rotation)

        return stats

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
        pass
