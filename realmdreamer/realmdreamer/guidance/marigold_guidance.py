# https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_guidance.py

import pdb
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cprint import *
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from torch import Tensor
from torchmetrics.functional.regression import pearson_corrcoef
from tqdm.auto import tqdm

from realmdreamer.extern.marigold.model.marigold_pipeline import \
    MarigoldPipeline


def align_tensors_by_median(tensor1, tensor2):

    tensor1_median = torch.median(tensor1)
    tensor2_median = torch.median(tensor2)

    return tensor1, tensor2 * (tensor1_median / (tensor2_median + 1e-5))


# Alex
def get_scale_translation(
    x: np.ndarray,  # (N,)
    y: np.ndarray,  # (N,)
):
    """
    Given two N dimensions tensors, compute a solution to a * x + b = y. Return a, b
    """

    x = x[:, None]  # (N, 1)
    x = np.concatenate([x, np.ones(shape=x.shape)], -1)  # (N, 2)
    psuedo = np.linalg.inv(x.T @ x) @ x.T

    scale, translation = (psuedo @ y[..., None]).squeeze()

    return scale, translation


def align_depths(
    source_depth: torch.Tensor,  # B 1 H W
    target_depth: torch.Tensor,  # B 1 H W
    mask: torch.Tensor = None,  # B 1 H W
    enforce_scale_positive=False,
):
    """
    Given two depth maps, align one with the other. If a mask is provided, will use only the values within the mask
    """

    assert (
        source_depth.shape == target_depth.shape
    ), f"Shape of input depths is not the same {source_depth.shape} vs {target_depth.shape}"

    assert mask.shape == source_depth.shape, f"Shape of mask is not same as depth {source_depth.shape} vs {mask.shape}"

    batch_size, _, _, _ = source_depth.shape

    output_depth = source_depth.clone()

    for i in range(batch_size):

        source_depth_i = source_depth[i].squeeze()  # H W
        target_depth_i = target_depth[i].squeeze()

        if mask is not None:
            mask_i = mask[i].squeeze()
            source_depth_i = source_depth_i[mask_i]
            target_depth_i = target_depth_i[mask_i]

        scale, translation = get_scale_translation(
            source_depth_i.flatten().detach().cpu().numpy(),
            target_depth_i.flatten().detach().cpu().numpy(),
        )

        # print(scale, "is scale")
        if scale < 0:

            if enforce_scale_positive:  # Exit early and return none if scale is meant to be positive
                return None

            print("Scale is negative!!!", scale)
            scale = 1
            # raise Warning("Scale should not be negative", scale)

        output_depth[i] = translation + scale * source_depth

    return output_depth  # B 1 H W


class MarigoldConfig:
    pretrained_model_name_or_path: str = "Bingxin/Marigold"
    enable_channels_last_format: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    weighting_strategy: str = "sds"

    anneal: bool = False


class MarigoldGuidance(nn.Module):

    def __init__(
        self,
        device: Union[torch.device, str],
        # full_precision=False,
        min_step_percent: float = 0.02,
        max_step_percent: float = 0.98,
    ) -> None:
        super().__init__()

        self.cfg = MarigoldConfig()
        self.device = device

        self.cfg.min_step_percent = min_step_percent
        self.cfg.max_step_percent = max_step_percent
        self.full_precision = True

        self.weights_dtype = torch.float32

        self.pipe = MarigoldPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
        ).to(self.device)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        # self.scheduler.alphas = self.scheduler.alphas.to(self.device)

        cprint.info(f"Loaded Marigold!")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,  # Float[Tensor, "..."],
        t,  # Float[Tensor, "..."],
        encoder_hidden_states,  # Float[Tensor, "..."],
    ):
        #  -> Float[Tensor, "..."]:
        input_dtype = latents.dtype

        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self,
        imgs,
        # Float[Tensor, "B 3 512 512"]
    ):
        # -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0

        h = self.pipe.vae.encoder(imgs.to(self.weights_dtype))
        moments = self.pipe.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # scale latent
        latents = mean * self.pipe.rgb_latent_scale_factor

        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_depths(
        self,
        depths,  # Float[Tensor, "B 1 512 512"]
    ):
        # -> Float[Tensor, "B 4 64 64"]:
        batch_size = depths.shape[0]

        input_dtype = depths.dtype

        # depth calculated as ((d - d_{0.2})/(d_{0.98} - d_{0.2}) - 0.5) * 2
        lower_quantile = torch.quantile(depths.view(batch_size, 1, -1), 0.02, dim=-1)
        upper_quantile = torch.quantile(depths.view(batch_size, 1, -1), 0.98, dim=-1)
        depth_normalized = (
            (depths - lower_quantile.view(batch_size, 1, 1, 1))
            / (upper_quantile - lower_quantile).view(batch_size, 1, 1, 1)
            - 0.5
        ) * 2

        # Repeat depth on 3 channels
        depth_normalized = depth_normalized.repeat(1, 3, 1, 1)

        h = self.pipe.vae.encoder(depth_normalized.to(self.weights_dtype))
        moments = self.pipe.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # scale latent
        latents = mean * self.pipe.depth_latent_scale_factor

        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_depth_latents(
        self,
        latents,  # Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        #  -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents

        z = self.pipe.vae.post_quant_conv(latents.to(self.weights_dtype))
        stacked = self.pipe.vae.decoder(z)

        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)

        return depth_mean.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_image_latents(
        self,
        latents,  # Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ):

        #  -> Float[Tensor, "B 1 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.pipe.rgb_latent_scale_factor * latents

        z = self.pipe.vae.post_quant_conv(latents.to(self.weights_dtype))
        stacked = self.pipe.vae.decoder(z)

        return stacked.to(input_dtype)

    def compute_grad_sds(
        self,
        rgb_latents,  # Float[Tensor, "B 8 64 64"],
        depth_latents,  # Float[Tensor, "B 8 64 64"],
        text_embeddings,  # Float[Tensor, "BB 77 768"],
        t,  # Int[Tensor, "B"],
    ):

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise only to the last 4 channels
            noise = torch.randn_like(depth_latents)  # TODO: use torch generator

            # print(latents.shape, noise.shape)

            depth_latents_noisy = self.scheduler.add_noise(depth_latents, noise, t)

            latents_noisy = torch.cat([rgb_latents, depth_latents_noisy], dim=1)

            # pred noise
            noise_pred = self.forward_unet(
                latents_noisy,
                t,
                encoder_hidden_states=text_embeddings,
            )

            # Compute the prev sample and one step estimate
            latent_dict = self.scheduler.step(noise_pred, t, depth_latents_noisy)
            latents_prev = latent_dict["prev_sample"]
            latents_one_step = latent_dict["pred_original_sample"]

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        depth_noise = noise
        grad = w * (noise_pred - depth_noise)
        return grad, w, (noise_pred - depth_noise), latents_one_step

    def __call__(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        depth,  # Float[Tensor, "B H W 1"],
        current_step_ratio: float,
        mask: Optional[Tensor] = None,
        rgb_as_latents=False,
        timesteps=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        self.scheduler.set_timesteps(self.num_train_timesteps, device=self.device)

        # assert rgb.min() >= 0.0 and rgb.max() <= 1.0

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        if mask is not None:
            mask = mask.permute(0, 3, 1, 2)

        # latents: Float[Tensor, "B 4 64 64"]
        start = time.time()
        if rgb_as_latents:

            rgb_latent = rgb_BCHW
            depth_latent = depth

            if mask is not None:
                mask = F.interpolate(mask.float(), (rgb_latent.shape[2], rgb_latent.shape[3]))
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW,
                (512, 512),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            depth_BCHW_512 = F.interpolate(depth, (512, 512), mode="nearest")

            # encode image and depth images
            rgb_latent = self.encode_images(rgb_BCHW_512)
            depth_latent = self.encode_depths(depth_BCHW_512)

            if mask is not None:
                mask = F.interpolate(mask.float(), (rgb_latent.shape[2], rgb_latent.shape[3]))

        end = time.time()

        # print(f"Latent time: {end-start}s")

        if self.pipe.empty_text_embed is None:
            self.pipe.encode_empty_text()

        batch_empty_text_embed = self.pipe.empty_text_embed.repeat(batch_size, 1, 1)  # [B, 2, 1024]

        if timesteps is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            if not self.cfg.anneal:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [batch_size],
                    dtype=torch.long,
                    device=self.device,
                )

                t = torch.tensor([t] * batch_size, dtype=torch.long, device=self.device)

            else:
                t = current_step_ratio * self.min_step + (1 - current_step_ratio) * self.max_step
                t = torch.tensor([t] * batch_size, dtype=torch.long, device=self.device)
        else:
            t = timesteps

        t = torch.tensor([t] * batch_size, dtype=torch.long, device=self.device)

        start = time.time()
        grad, w, noise_diff, one_step_estimate = self.compute_grad_sds(
            rgb_latent, depth_latent, batch_empty_text_embed, t
        )
        end = time.time()

        with torch.no_grad():
            one_step_estimate_decoded = self.decode_depth_latents(one_step_estimate)

        depth_img_loss = F.mse_loss(one_step_estimate_decoded.detach(), depth, reduction="sum") / batch_size

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (depth_latent - grad).detach()
        # target = (latents - grad)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        # loss_sds = 0.5 * F.mse_loss(depth_latent, target, reduction="sum") / batch_size
        loss_sds = depth_img_loss

        assert not torch.isnan(loss_sds), "loss_sds is NaN"

        return {
            "loss_sds": loss_sds,
        }, {
            "timesteps": t,
            "grad": grad,
            "w": w,
            "noise_diff": noise_diff,
            "one_step_pred": one_step_estimate_decoded + 1,
        }

    # Sample from the depth model
    @torch.no_grad()
    def sample(self, rgb_in, depth_in, strength, num_inference_steps=10):  # B C H W  # B 1 H W

        batch_size = rgb_in.shape[0]
        device = self.device

        # Encode image
        rgb_latent = self.encode_images(rgb_in)

        # Initial depth map (noise)
        if depth_in is not None:
            depth_latent = self.encode_depths(depth_in)
        else:
            depth_latent = torch.randn((batch_size, 4, 64, 64), device=device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        try:
            timesteps, num_steps, init_timestep = self.get_timesteps(
                self.scheduler,
                num_inference_steps,  # for full run, use num_train_timesteps
                strength,
                depth_latent.device,
            )

            # Add noise corresponding to a strength to the latent
            noise = torch.randn_like(depth_latent)
            latents = self.scheduler.add_noise(depth_latent, noise, timesteps[0])
        except:

            print("failed to get timesteps, using default")

            timesteps, num_steps, init_timestep = self.get_timesteps(
                self.scheduler,
                num_inference_steps,  # for full run, use num_train_timesteps
                0.2,
                depth_latent.device,
            )

            # Add noise corresponding to a strength to the latent
            noise = torch.randn_like(depth_latent)
            latents = self.scheduler.add_noise(depth_latent, noise, timesteps[0])

        if self.pipe.empty_text_embed is None:
            self.pipe.encode_empty_text()

        batch_empty_text_embed = self.pipe.empty_text_embed.repeat(batch_size, 1, 1)  # [B, 2, 1024]

        # Denoising loop
        iterable = tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            leave=False,
            desc=" " * 4 + "Marigold denoising",
        )

        print("starting denoising loop")
        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, latents], dim=1)  # this order is important

            # predict the noise residual
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dict = self.scheduler.step(noise_pred, t, latents)

            latents = latents_dict["prev_sample"]
            latents_0 = latents_dict["pred_original_sample"]

            decoded_0 = self.decode_depth_latents(latents_0)

            # plt.imshow(decoded_0[0].cpu().numpy().transpose(1, 2, 0))
            # plt.colorbar()
            # plt.title(f"{i} - timestep {t}")
            # plt.show()

        depth = self.decode_depth_latents(latents)

        # normalize depth
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        # # clip prediction
        # depth = torch.clip(depth, -1.0, 1.0)
        # # shift to [0, 1]
        # depth = depth * 2.0 - 1.0

        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        return depth

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, scheduler, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]

        return timesteps, num_inference_steps - t_start, init_timestep

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        pass

    @torch.no_grad()
    def pred_one_step(self, rgb_in, depth_in=None, strength=1.0, pbar=False):  # B C H W

        num_inference_steps = 10

        batch_size = rgb_in.shape[0]
        device = self.device

        # Encode image
        rgb_latent = self.encode_images(rgb_in)

        # Initial depth map (noise)
        if depth_in is not None:
            depth_latent = self.encode_depths(depth_in)
        else:
            depth_latent = torch.randn((batch_size, 4, 64, 64), device=device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        try:
            timesteps, num_steps, init_timestep = self.get_timesteps(
                self.scheduler,
                num_inference_steps,  # for full run, use num_train_timesteps
                strength,
                depth_latent.device,
            )

            # Add noise corresponding to a strength to the latent

            if depth_in is not None:
                noise = torch.randn_like(depth_latent)
                latents = self.scheduler.add_noise(depth_latent, noise, timesteps[0])
            else:
                noise = depth_latent
                latents = depth_latent
        except:

            print("failed to get timesteps, using default")
            raise Exception

            timesteps, num_steps, init_timestep = self.get_timesteps(
                self.scheduler,
                num_inference_steps,  # for full run, use num_train_timesteps
                0.2,
                depth_latent.device,
            )

            # Add noise corresponding to a strength to the latent
            noise = torch.randn_like(depth_latent)
            latents = self.scheduler.add_noise(depth_latent, noise, timesteps[0])

        if self.pipe.empty_text_embed is None:
            self.pipe.encode_empty_text()

        batch_empty_text_embed = self.pipe.empty_text_embed.repeat(batch_size, 1, 1)  # [B, 2, 1024]

        # Denoising loop
        if pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, latents], dim=1)  # this order is important

            # predict the noise residual
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dict = self.scheduler.step(noise_pred, t, latents)

            latents = latents_dict["prev_sample"]
            latents_0 = latents_dict["pred_original_sample"]

            break

        depth = self.decode_depth_latents(latents_0)

        # normalize depth
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        return depth


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value
