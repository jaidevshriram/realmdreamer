# https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_guidance.py

import pdb
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union

import kornia
import matplotlib.pyplot as plt
import numpy as np
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from cprint import *
from diffusers import (DDIMInverseScheduler, DDIMScheduler, DDPMScheduler,
                       StableDiffusionInpaintPipeline)
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torch import Tensor
from tqdm.auto import tqdm

from utils.diffusers import custom_step


class SDInpaintingConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-inpainting"
    enable_memory_efficient_attention: bool = True
    enable_channels_last_format: bool = False
    guidance_scale: float = 7.5
    img_guidance_scale: float = 1.2
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    nfsd: bool = False
    weighting_strategy: str = "sds"  # sds / uniform
    rescale: bool = False
    recon_std_rescale: float = 0.5

    invert: bool = False  # Whether to obtain the noisy latent using inverse diffusion
    inversion_method: str = "ddim"  # 'ddim' or 'ddpm '(https://inbarhub.github.io/DDPM_inversion/)

    num_steps_sample: int = 10

    grad_clip: bool = True
    grad_clip_val: float = 1
    anneal: bool = False
    prolific_anneal: bool = False  # Drop the max step to 50% after 50% of the steps
    anneal_ratio_start: float = 0.75  # Start annealing after 75% of the steps


class StableDiffusionInpaintingGuidance(nn.Module):

    def __init__(
        self,
        device: Union[torch.device, str],
        model_path: str = None,
        full_precision=False,
        guidance_scale: float = 7.5,
        img_guidance_scale: float = 1.0,
        min_step_percent: float = 0.02,
        max_step_percent: float = 0.98,
        anneal: bool = False,
        prolific_anneal: bool = False,
        invert_ddim: bool = False,
        num_steps_sample: int = 10,
    ) -> None:
        super().__init__()

        self.cfg = SDInpaintingConfig()

        # Override config
        self.cfg.min_step_percent = min_step_percent
        self.cfg.max_step_percent = max_step_percent
        self.cfg.guidance_scale = guidance_scale
        self.cfg.img_guidance_scale = img_guidance_scale
        self.cfg.anneal = anneal
        self.cfg.prolific_anneal = prolific_anneal
        self.cfg.invert = invert_ddim
        self.cfg.num_steps_sample = num_steps_sample

        self.device = device
        self.full_precision = full_precision

        if model_path is not None:
            self.cfg.pretrained_model_name_or_path = model_path

        self.weights_dtype = torch.float32 if self.full_precision else torch.float16

        pipe_kwargs = {
            # "tokenizer": None, # We use the pipe's tokenizer
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                print("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)

        # Create model
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.scheduler = DDIMScheduler.from_config(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.ddim_inverse_scheduler = DDIMInverseScheduler.from_config(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val = self.cfg.grad_clip_val

        self.pipe.set_progress_bar_config(disable=True)
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)

        self.text_embeddings = None

        self.scheduler.set_timesteps(self.num_train_timesteps)

        cprint.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def multi_step(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        og_rgb,  # Float[Tensor, "B H W C"],
        mask,  # Bool[Tensor, "B H W 1"],
        prompt: str,
        current_step_ratio,
        rgb_as_latents=False,
        mask_grad=True,
        **kwargs,
    ):

        batch_size = rgb.shape[0]

        if self.cfg.anneal:

            if self.cfg.prolific_anneal:
                if current_step_ratio >= 0.5:
                    self.max_step = int(self.num_train_timesteps * 0.5)

                # timestep ~ U(0.02, max noise) to avoid very high/low noise level
                t = np.random.randint(low=self.min_step, high=self.max_step + 1)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            else:

                t = current_step_ratio * self.min_step + (1 - current_step_ratio) * self.max_step
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        with torch.no_grad():
            inpainted_image, _ = self.sample(
                rgb=rgb,
                mask=mask,
                prompt=prompt,
                strength=t / self.num_train_timesteps,
            )

            og_rgb_BCHW = og_rgb.permute(0, 3, 1, 2)

        w = self.get_weighting(t)

        l1_img_loss = F.l1_loss(inpainted_image, og_rgb_BCHW, reduction="mean")
        l2_img_loss = F.mse_loss(inpainted_image, og_rgb_BCHW, reduction="sum") / batch_size
        lpips_img_loss = self.lpips(inpainted_image * 2 - 1, og_rgb_BCHW * 2 - 1)

        loss_nfsd = 1000 * (lpips_img_loss + l2_img_loss)

        return {
            "loss_sds": loss_nfsd,
        }, {
            "timesteps": t,
            "grad": torch.zeros_like(rgb),
            "w": w,
            "multi_step_pred": inpainted_image,
        }

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,  # Float[Tensor, "..."],
        t,  # Float[Tensor, "..."],
        encoder_hidden_states,  # Float[Tensor, "..."],
    ):
        #  -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        sample = self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            return_dict=False,
        )[0]

        return sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self,
        imgs,
        # Float[Tensor, "B 3 512 512"]
    ):
        # -> Float[Tensor, "B 4 64 64"]:

        # Ensure that images is in the correct shape

        imgs = F.interpolate(imgs, (512, 512), mode="bilinear", align_corners=False)

        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents,  # Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        #  -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def get_text_embeddings(self, prompt, batch_size):
        """
        Get text embeddings for a given prompt - per batch size

        Returns:

        - text_embeddings: Float[Tensor, "BB 77 768"] - The text embeddings for the prompt (conditional, unconditional/negative text)
        """

        text_embeddings = self.pipe._encode_prompt(
            prompt=prompt,
            negative_prompt="ugly, blurry, text, pixelated obscure, unnatural colors, poor lighting, dull, cropped, lowres, low quality",
            device=self.device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # Split into cond and uncond
        text_embeddings_uncond, text_embeddings_cond = text_embeddings.chunk(2)

        # Combine it back into cond and unconcd
        text_embeddings = torch.cat([text_embeddings_cond, text_embeddings_uncond], dim=0)

        return text_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(
        self,
        scheduler,
        num_inference_steps,
        strength,
        device,
        fixed_inference_count=False,
    ):
        """
        Get the timesteps for the diffusion model based on the strength of noise
        """

        if not fixed_inference_count:

            # get the original timestep using init_timestep
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = scheduler.timesteps[t_start * scheduler.order :]

            return timesteps, num_inference_steps - t_start, init_timestep

        else:

            # get the original timestep using init_timestep
            init_timestep = (
                min(int(num_inference_steps * strength), num_inference_steps)
                * self.num_train_timesteps
                / num_inference_steps
            )

            timesteps = torch.round(torch.arange(init_timestep, 0, -init_timestep / num_inference_steps)).to(
                scheduler.timesteps.dtype
            )

            return timesteps, num_inference_steps, init_timestep

    @torch.no_grad()
    def sample(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        mask,  # Bool[Tensor, "B H W 1"],
        prompt: str,
        strength: float,
        num_inference_steps=25,
        img_cfg=1.0,
        fixed_inference_count=False,
    ):
        """

        Sample from the inpainting model - given an RGB image and a mask, return the inpainted image

        Args:
        - rgb: Float[Tensor, "B H W C"] - The RGB image to inpaint
        - mask: Bool[Tensor, "B H W 1"] - The mask for the image
        - prompt: str - The prompt for the image
        - strength: float - The strength of the noise added
        - num_inference_steps: int - The number of inference steps to run
        - img_cfg: float - The image guidance scale (if img_cfg=1.0, then it defaluts to the regular inpainting model sampling step)
        - fixed_inference_count: bool - If True, then the number of inference steps is fixed, else it is based on the strength

        """

        batch_size = rgb.shape[0]
        guidance_scale = self.cfg.guidance_scale

        # Text embeddings
        text_embeddings_cond, text_embeddings_uncond = self.get_text_embeddings(prompt, batch_size)

        text_embeddings = torch.stack(
            [
                text_embeddings_uncond,  # (for no mask, no text)
                text_embeddings_uncond,  # (for mask, no text)
                text_embeddings_cond,  #  (for mask, text)
            ],
            dim=0,
        )

        # The goal of this is to take a RGB image, add noise to it, and then denoise it to obtain an inpainted image
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        # Get image latents, mask, and masked latents
        latents = self.encode_images(rgb_BCHW)
        masked_img = (rgb_BCHW * 2 - 1) * (mask < 0.5)
        masked_img = (masked_img * 0.5 + 0.5).clamp(0, 1)
        masked_latent = self.encode_images(masked_img)

        masked_empty_img = torch.ones_like(masked_img) * 0.5
        masked_empty_img_latent = self.encode_images(masked_empty_img)

        # Combine the masks
        mask_64 = F.interpolate(mask.float(), (64, 64))
        mask_64_00 = torch.ones_like(mask_64)
        masks = torch.cat(
            [
                mask_64_00,
                mask_64,
                mask_64,
            ],
            dim=0,
        )  # The masks are (fulll mask, mask, mask)

        masked_latents = torch.cat(
            [
                masked_empty_img_latent,
                masked_latent,
                masked_latent,
            ],
            dim=0,
        )  # The masked latents corresponding to the (full mask, mask, mask)

        self.scheduler.config.timestep_spacing = (
            "trailing"  # Supposed to be better as per https://arxiv.org/pdf/2305.08891.pdf
        )
        self.scheduler.set_timesteps(num_inference_steps)
        self.ddim_inverse_scheduler.set_timesteps(num_inference_steps)

        timesteps, num_steps, init_timestep = self.get_timesteps(
            self.scheduler,
            num_inference_steps,  # for full run, use num_train_timesteps
            strength,
            latents.device,
            fixed_inference_count=fixed_inference_count,
        )

        if not self.cfg.invert:
            # Add noise corresponding to a strength to the latent
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        else:
            latents = self.invert(
                rgb_BCHW,
                rgb_BCHW,
                mask,
                prompt,
                cfg=0.0,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
            )

        if strength == 1.0:
            noise = torch.randn_like(latents)
            latents = noise  # If strength is one, start from pure noise

        # Get timesteps required for denoising
        one_step_pred_0 = None

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False, desc="Sampling"):

            latent_model_input = torch.cat([latents] * 3)

            latent_model_input = torch.cat([latent_model_input, masks, masked_latents], dim=1)

            noise_pred = self.forward_unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings.to(self.weights_dtype),
            )

            noise_pred_00, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_00
                + img_cfg * (noise_pred_uncond - noise_pred_00)
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
            )

            # compute the previous noisy sample x_t -> x_t-1
            if i != len(timesteps) - 1:
                latent_dict = custom_step(self.scheduler, noise_pred, t, timesteps[i + 1], latents)
            else:
                latent_dict = custom_step(
                    self.scheduler, noise_pred, t, -1, latents
                )  # will automatically get set to final step

            latents = latent_dict["prev_sample"]
            latents_0 = latent_dict["pred_original_sample"]

            # pred_images = self.decode_latents(latents_0).detach()
            # one_step_pred_0 = pred_images

            # Plot the images
            # plt.imshow(pred_images[0].cpu().permute(1,2,0).clamp(0,1))
            # plt.title("One step at" + str(i))
            # plt.show()

        images = self.decode_latents(latents_0)
        one_step_pred_0 = images.detach()
        return images, one_step_pred_0

    @torch.no_grad()
    def sample_from_latent(self, latent, og_img, mask, prompt, cfg, num_inference_steps=25, timesteps=None):
        """

        Given a particular noise latent, sample an image from it using the prompt and cfg for the current diffusion model

        Args:
        - latent: Float[Tensor, "B 4 64 64"] - This is the latent to sample from
        - og_img: Float[Tensor, "B 3 512 512"] - The original image
        - mask: Bool[Tensor, "B 1 512 512"]
        - prompt: str
        - cfg: float
        - num_inference_steps: int

        """

        batch_size = latent.shape[0]

        # image = self.decode_latents(latent)
        masked_latent = self.get_masked_latents(og_img, og_img, mask)[2]
        mask_64 = F.interpolate(mask.float(), (64, 64))

        text_embeddings = self.get_text_embeddings(prompt, batch_size)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps if timesteps is None else timesteps

        for i, t in tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            leave=True,
        ):

            t_b1 = t.repeat(batch_size).to(latent.device)  # [batch_size]
            t_b2 = t.repeat(batch_size * 2).to(latent.device)  # [batch_size * 2] - for unet

            latent_input = torch.cat([latent] * 2, dim=0)
            mask_input = torch.cat([mask_64] * 2, dim=0)
            masked_latent_input = torch.cat([masked_latent] * 2, dim=0)

            latent_input = torch.cat([latent_input, mask_input, masked_latent_input], dim=1)

            noise_pred = self.forward_unet(
                latent_input,
                t_b2,
                encoder_hidden_states=text_embeddings,
            )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

            latent = self.scheduler.step(noise_pred, t, latent)["prev_sample"]

        return latent

    @torch.no_grad()
    def invert(
        self,
        image,
        og_image,
        mask,
        prompt,
        cfg,
        img_cfg=0.0,
        num_inference_steps=25,
        timesteps=None,
    ):
        """

        Invert the image using the prompt, cfg for the current diffusion model using DDIM inversion
        https://github.com/inbarhub/DDPM_inversion/blob/main/ddm_inversion/ddim_inversion.py

        Args:
        - image: Float[Tensor, "B 3 512 512"]
        - mask: Bool[Tensor, "B 1 512 512"]
        - prompt: str
        - cfg: float
        - img_cfg: float

        """

        batch_size = image.shape[0]

        # Text embeddings
        text_embeddings_cond, text_embeddings_uncond = self.get_text_embeddings(prompt, batch_size)

        text_embeddings = torch.stack(
            [
                text_embeddings_uncond,  # (for no mask, no text)
                text_embeddings_uncond,  # (for mask, no text)
                text_embeddings_cond,  #  (for mask, text)
            ],
            dim=0,
        )

        latent = self.encode_images(image)
        masked_latent = self.get_masked_latents(image, og_image, mask)[2]
        mask_64 = F.interpolate(mask.float(), (64, 64))

        masked_empty_img = torch.ones_like(image) * 0.5
        masked_empty_img_latent = self.encode_images(masked_empty_img)
        mask_64_00 = torch.ones_like(mask_64)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps if timesteps is None else timesteps

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False, desc="Inverting"):

            t = timesteps[len(timesteps) - i - 1]
            t_b1 = t.repeat(batch_size).to(image.device)  # [batch_size]
            t_b3 = t.repeat(batch_size * 3).to(image.device)  # [batch_size * 3] - for unet

            latent_input = torch.cat([latent] * 3, dim=0)
            mask_input = torch.cat([mask_64_00, mask_64, mask_64], dim=0)
            masked_latent_input = torch.cat([masked_empty_img_latent, masked_latent, masked_latent], dim=0)

            latent_input = torch.cat([latent_input, mask_input, masked_latent_input], dim=1)

            noise_pred = self.forward_unet(
                latent_input,
                t_b3,
                encoder_hidden_states=text_embeddings,
            )

            noise_pred_00, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_00
                + img_cfg * (noise_pred_uncond - noise_pred_00)
                + cfg * (noise_pred_text - noise_pred_uncond)
            )

            latent = next_step_ddim_invert(self.scheduler, noise_pred, t, latent)

            # decoded = self.decode_latents(latent)
            # import matplotlib.pyplot as plt
            # plt.imshow(decoded[0].cpu().permute(1,2,0).clamp(0,1))
            # plt.title("One step at" + str(i))
            # plt.show()

        return latent

    def get_masked_latents(self, rgb_BCHW, og_rgb_BCHW, mask):

        if rgb_BCHW.shape[2] != 512:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW,
                (512, 512),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            og_rgb_BCHW_512 = F.interpolate(
                og_rgb_BCHW,
                (512, 512),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            mask_512 = F.interpolate(mask.float(), (512, 512), mode="nearest")
        else:
            rgb_BCHW_512 = rgb_BCHW
            og_rgb_BCHW_512 = og_rgb_BCHW
            mask_512 = mask.float()

        masked_og_image = (og_rgb_BCHW_512 * 2 - 1) * (mask_512 < 0.5)
        # masked_og_image = og_rgb_BCHW

        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW_512)
        latents_og = self.encode_images(((masked_og_image * 0.5) + 0.5).clamp(0, 1))

        mask = F.interpolate(mask.float(), (latents.shape[2], latents.shape[3]), mode="nearest")

        # masked_image_latent = latents_og.clone() * (mask.float() < 0.5)
        masked_image_latent = latents_og.detach()

        return latents, mask, masked_image_latent

    def get_weighting(self, t):

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "inverted_sds":
            w = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = torch.ones_like(t).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t])).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "fantasia3d_type2":
            w = 1 / (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        return w


def next_step_ddim_invert(
    scheduler,
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
):
    """
    Takes a DDIM step and returns the inverted sample

    Args:
    - model: StableDiffusionPipeline
    - scheduler: DDIMScheduler
    - model_output: Union[torch.FloatTensor, np.ndarray] - the noise predicted from the model
    - timestep: int - the current timestep
    - sample: Union[torch.FloatTensor, np.ndarray] - the current sample/latent

    """

    timestep, next_timestep = (
        min(
            timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample


def parse_version(ver: str):
    return version.parse(ver)
