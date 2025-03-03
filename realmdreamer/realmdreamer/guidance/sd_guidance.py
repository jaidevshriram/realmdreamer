# https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_guidance.py

import pdb
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cprint import *
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm.auto import tqdm

sys.path.append("/mnt/data/temp/Portal")
from utils.diffusers import custom_step

# SD_SOURCE = "runwayml/stable-diffusion-v1-5"


class SDConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    enable_memory_efficient_attention: bool = True
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False
    guidance_scale: float = 7.5
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    use_sjc: bool = False
    var_red: bool = True
    weighting_strategy: str = "sds"
    rescale: bool = False
    recon_std_rescale: float = 0.5

    num_steps_sample: int = 10
    fixed_num_steps: bool = False  # Whether to use a fixed number of steps regardless of the strength

    invert: bool = False  # Whether to obtain the noisy latent using inverse diffusion
    inversion_method: str = "ddim"  # 'ddim' or 'ddpm '(https://inbarhub.github.io/DDPM_inversion/) or 'pseudo_ddim'

    grad_clip: bool = True
    grad_clip_val: float = 1
    anneal: bool = False
    prolific_anneal: bool = False  # Drop the max step to 50% after 50% of the steps


class StableDiffusionGuidance(nn.Module):

    def __init__(
        self,
        device: Union[torch.device, str],
        full_precision=False,
        model_path: Optional[str] = None,
        guidance_scale: float = 7.5,
        min_step_percent: float = 0.02,
        max_step_percent: float = 0.98,
        anneal: bool = False,
        prolific_anneal: bool = False,
        invert_ddim: bool = False,
        num_steps_sample: int = 10,
        ddim_invert_method: str = "ddim",
        fixed_num_steps: bool = False,
    ) -> None:
        super().__init__()

        self.cfg = SDConfig()
        self.device = device

        self.cfg.min_step_percent = min_step_percent
        self.cfg.max_step_percent = max_step_percent
        self.cfg.guidance_scale = guidance_scale
        self.full_precision = full_precision
        self.grad_clip_val = self.cfg.grad_clip_val
        self.cfg.invert = invert_ddim
        self.cfg.num_steps_sample = num_steps_sample
        self.cfg.inversion_method = ddim_invert_method
        self.cfg.fixed_num_steps = fixed_num_steps

        self.weights_dtype = torch.float32 if self.full_precision else torch.float16

        if model_path is not None:
            self.cfg.pretrained_model_name_or_path = model_path

        pipe_kwargs = {
            # "tokenizer": None, # We use the pipe's tokenizer
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if not is_xformers_available():
                cprint.warn("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.betas = self.scheduler.betas.to(self.device)

        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

        self.grad_clip_val = None

        cprint.info(f"Loaded Stable Diffusion!")

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

    def compute_grad_sds(
        self,
        latents,  # Float[Tensor, "B 4 64 64"],
        text_embeddings,  # Float[Tensor, "BB 77 768"],
        t,  # Int[Tensor, "B"],
    ):

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = self.get_weighting(t)

        grad = w * (noise_pred - noise)
        return grad, w, (noise_pred - noise)

    def compute_grad_sjc(
        self,
        latents,  # Float[Tensor, "B 4 64 64"],
        text_embeddings,  # Float[Tensor, "BB 77 768"],
        t,  # Int[Tensor, "B"],
    ):
        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma**2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            Ds = zs - sigma * noise_pred

            if self.cfg.var_red:
                grad = -(Ds - y) / sigma
            else:
                grad = -(Ds - zs) / sigma

        return grad

    @torch.cuda.amp.autocast(enabled=False)
    def compute_grad_nsfd(
        self,
        latents,  # Float[Tensor, "B 4 64 64"],
        text_embeddings,  # Float[Tensor, "BB 77 768"],
        t,  # Int[Tensor, "B"],
    ):

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(self.num_train_timesteps)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            tt = torch.cat([t] * 3)

            # Compute the noise for the unconditional, conditional, and negative text
            noise_pred = self.forward_unet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
            )

        noise_pred_uncond, noise_pred_y, noise_pred_neg = noise_pred.chunk(3)

        if t[0].item() < 200:
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_y - noise_pred_uncond)
        else:
            noise_pred = (noise_pred_uncond - noise_pred_neg) + self.cfg.guidance_scale * (
                noise_pred_y - noise_pred_uncond
            )

        w = self.get_weighting(t)
        grad = w * (noise_pred)

        noise_pred_one_step = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_y - noise_pred_uncond)

        if batch_size == 1:
            # compute the previous noisy sample x_t -> x_t-1
            latent_dict = self.scheduler.step(noise_pred_one_step, t, latents_noisy)
            latents = latent_dict["prev_sample"]
            latents_0 = latent_dict["pred_original_sample"]
        else:
            latents_0 = torch.zeros_like(latents_noisy)

            for batch_idx in range(batch_size):

                # compute the previous noisy sample x_t -> x_t-1
                latent_dict = self.scheduler.step(
                    noise_pred_one_step[batch_idx],
                    t[batch_idx],
                    latents_noisy[batch_idx],
                )
                latents[batch_idx] = latent_dict["prev_sample"]
                latents_0[batch_idx] = latent_dict["pred_original_sample"]

        if self.cfg.rescale:

            latents_recon_nocfg = torch.zeros_like(latents)
            for batch_idx in range(batch_size):

                # compute the previous noisy sample x_t -> x_t-1 but using the Positive Text noise only
                latent_dict = self.scheduler.step(noise_pred_y[batch_idx], t[batch_idx], latents_noisy[batch_idx])
                latents_recon_nocfg[batch_idx] = latent_dict["pred_original_sample"]

            factor = (latents_recon_nocfg.std([1, 2, 3], keepdim=True) + 1e-8) / (
                latents_0.std([1, 2, 3], keepdim=True) + 1e-8
            )

            latents_0_adjust = latents_0.clone() * factor
            latents_0 = self.cfg.recon_std_rescale * latents_0_adjust + (1 - self.cfg.recon_std_rescale) * latents_0

        # Decode the predictions to get the RGB prediction
        with torch.no_grad():
            pred_images = self.decode_latents(latents_0).detach()

        return grad, w, noise_pred, noise, pred_images

    @torch.cuda.amp.autocast(enabled=False)
    def nfsd_loss(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        prompt: str,
        current_step_ratio,
        mask: Optional[Tensor] = None,
        rgb_as_latents=False,
        **kwargs,
    ):

        self.scheduler.set_timesteps(self.num_train_timesteps)

        batch_size = rgb.shape[0]

        if rgb_as_latents:
            rgb_BCHW = rgb.permute(0, 3, 1, 2)
            latents = rgb_BCHW
        else:
            # The goal of this is to take a RGB image, add noise to it, and then denoise it to obtain an inpainted image
            rgb_BCHW = rgb.permute(0, 3, 1, 2)

            latents = self.encode_images(rgb_BCHW)

        # Get text embeddings
        text_embeddings = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # Split into cond and uncond (note that diffusers returns uncond, cond in v 19.1 at least)
        text_embeddings_uncond, text_embeddings_cond = text_embeddings.chunk(2)

        text_embeddings_neg = self.pipe._encode_prompt(
            prompt="unrealistic, saturated, blurry, low quality, out of focus, ugly, dull, dark, low-resolution, gloomy",
            # prompt="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy",
            device=self.device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        _, text_embeddings_neg = text_embeddings_neg.chunk(2)

        # Combine it back into uncond, cond
        text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond, text_embeddings_neg], dim=0)

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

        grad, w, noise_pred, noise, pred_images = self.compute_grad_nsfd(
            latents=latents, text_embeddings=text_embeddings, t=t
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.cfg.grad_clip and self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # target = (latents - grad)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        latent_loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        if not rgb_as_latents:
            # Resize predicted image to shape of input images
            pred_images = F.interpolate(
                pred_images,
                (rgb_BCHW.shape[2], rgb_BCHW.shape[3]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        loss_nfsd = latent_loss

        return {
            "loss_sds": loss_nfsd,
        }, {
            "timesteps": t,
            "grad": grad,
            "w": w,
            "one_step_pred": pred_images,
        }

    @torch.cuda.amp.autocast(enabled=False)
    def multi_step(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        mask,  # Bool[Tensor, "B H W 1"],
        prompt: str,
        current_step_ratio,
        rgb_as_latents=False,
        latent_strat="correct",  # 'correct' or 'hack' - in hack, we simply mask the latent
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
                [1],
                dtype=torch.long,
                device=self.device,
            )

            t = t.repeat(batch_size)

        with torch.no_grad():
            inpainted_image, _ = self.sample(
                rgb=rgb,
                mask=mask,
                prompt=prompt,
                strength=t / self.num_train_timesteps,
                num_inference_steps=self.cfg.num_steps_sample,
                fixed_inference_count=self.cfg.fixed_num_steps,
            )

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        w = self.get_weighting(t)

        inpainted_image_latent = self.encode_images(inpainted_image)
        og_latents = self.encode_images(rgb_BCHW)

        # Latent difference is
        latent_loss = (
            w[0].item() * F.mse_loss(inpainted_image_latent.detach(), og_latents, reduction="sum") / batch_size
        )

        loss_nfsd = latent_loss

        return {
            "loss_sds": loss_nfsd,
        }, {
            "timesteps": t,
            "grad": torch.zeros_like(rgb),
            "w": w,
            "multi_step_pred": inpainted_image,
        }

    def __call__(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        prompt: str,
        current_step_ratio: float,
        mask: Optional[Tensor] = None,
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        # assert rgb.min() >= 0.0 and rgb.max() <= 1.0

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        # latents: Float[Tensor, "B 4 64 64"]
        latents = None
        if rgb_as_latents:
            latents = F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False)

            mask = F.interpolate(mask.float(), (latents.shape[2], latents.shape[3]))
        else:
            rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)

            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

            mask = F.interpolate(mask.float(), (latents.shape[2], latents.shape[3]))

        text_embeddings = self.pipe._encode_prompt(
            prompt,
            self.device,
            batch_size,
            True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # Split into cond and uncond
        text_embeddings_cond, text_embeddings_uncond = text_embeddings.chunk(2)

        # Combine it back into uncond, cond
        text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond], dim=0)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if not self.cfg.anneal:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
        else:
            t = current_step_ratio * self.min_step + (1 - current_step_ratio) * self.max_step
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

        if self.cfg.use_sjc:
            grad = self.compute_grad_sjc(latents, text_embeddings, t)
        else:
            grad, w, noise_diff = self.compute_grad_sds(latents, text_embeddings, t)

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        if mask is not None:
            grad = grad * mask

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # target = (latents - grad)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return {
            "loss_sds": loss_sds,
        }, {
            "timesteps": t,
            "grad": grad,
            "w": w,
            "noise_diff": noise_diff,
        }

    @torch.no_grad()
    def sample(
        self,
        rgb,  # Float[Tensor, "B H W C"],
        mask,  # Bool[Tensor, "B H W 1"],
        prompt: str,
        strength: float,
        num_inference_steps=25,
        fixed_inference_count=False,
    ):

        batch_size = rgb.shape[0]
        guidance_scale = self.cfg.guidance_scale

        # Text embeddings
        text_embeddings = self.get_text_embeddings(prompt, batch_size).to(self.weights_dtype)
        # text_embeddings_cond, text_embeddings_uncond = text_embeddings.chunk(2)

        # The goal of this is to take a RGB image, add noise to it, and then denoise it to obtain an inpainted image
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        latents = self.encode_images(rgb_BCHW)

        self.scheduler.config.timestep_spacing = (
            "leading"  # 'trailing' is supposed to be better as per https://arxiv.org/pdf/2305.08891.pdf
        )
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps, num_steps, init_timestep = self.get_timesteps(
            self.scheduler,
            num_inference_steps,  # for full run, use num_train_timesteps
            strength,
            latents.device,
            fixed_inference_count=fixed_inference_count,
        )

        if self.cfg.invert:
            if self.cfg.inversion_method == "ddim" or "ddim_1":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=1.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "pseudo_ddim":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=1.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps[:-1],
                )

                # Add some more noise to the latents (t-1 -> t)
                noise = torch.randn_like(latents)
                beta_t = self.betas[timesteps[-1].int().cpu()]
                latents = torch.sqrt(1 - beta_t) * latents + beta_t * noise

            elif self.cfg.inversion_method == "ddim_0":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=0.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "ddim_2":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=2.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "ddim_3":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=3.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "ddim_4":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=4.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "ddim_5":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=5.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "ddim_6":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=6.0,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            elif self.cfg.inversion_method == "ddim_75":
                latents = self.invert(
                    rgb_BCHW,
                    prompt,
                    cfg=7.5,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                )
            else:
                raise ValueError(f"Unknown inversion method: {self.cfg.inversion_method}")
        else:
            # Add noise corresponding to a strength to the latent
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        # If strength is 1.0, then we sample from pure noise
        if (type(strength) == torch.tensor and strength[0] == 1.0) or (type(strength) == float and strength == 1.0):
            latents = torch.randn_like(latents)

        # Get timesteps required for denoising
        one_step_pred_0 = None

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False, desc="Sampling"):

            latent_model_input = torch.cat([latents] * 2)

            t_b1 = t.repeat(batch_size).to(latents.device)
            t_b2 = t.repeat(batch_size * 2).to(latents.device)

            # print(t)

            noise_pred = self.forward_unet(
                latent_model_input,
                t_b2,
                encoder_hidden_states=text_embeddings,
            )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # latent_dict = self.scheduler.step(noise_pred, t, latents)

            # compute the previous noisy sample x_t -> x_t-1
            if i != len(timesteps) - 1:
                latent_dict = custom_step(self.scheduler, noise_pred, t, timesteps[i + 1], latents)
            else:
                latent_dict = custom_step(
                    self.scheduler, noise_pred, t, -1, latents
                )  # will automatically get set to final step

            latents = latent_dict["prev_sample"]
            latents_0 = latent_dict["pred_original_sample"]

        images = self.decode_latents(latents_0).detach()
        one_step_pred_0 = images.clone()
        return images, one_step_pred_0

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

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

    def get_text_embeddings(self, prompt, batch_size):
        """
        Get text embeddings for a given prompt - per batch size

        Returns:
        - text_embeddings: Float[Tensor, "BB 77 768"] - (conditional, unconditional)
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

        self.scheduler.set_timesteps(num_inference_steps)

        if not fixed_inference_count:

            if type(strength) == torch.Tensor:
                strength = strength[0]

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
    def sample_from_latent(self, latent, prompt, cfg, num_inference_steps=25, timesteps=None):
        """

        Given a particular noise latent, sample an image from it using the prompt and cfg for the current diffusion model

        Args:
        - latent: Float[Tensor, "B 4 64 64"] - This is the latent to sample from
        - prompt: str
        - cfg: float
        - num_inference_steps: int

        """

        batch_size = latent.shape[0]

        text_embeddings = self.get_text_embeddings(prompt, batch_size)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps if timesteps is None else timesteps

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False):

            # t = timesteps[len(timesteps) - i - 1]
            t_b1 = t.repeat(batch_size).to(latent.device)  # [batch_size]
            t_b2 = t.repeat(batch_size * 2).to(latent.device)  # [batch_size * 2] - for unet

            latent_input = torch.cat([latent] * 2, dim=0)

            noise_pred = self.forward_unet(
                latent_input,
                t_b2,
                encoder_hidden_states=text_embeddings,
            )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

            latent = self.scheduler.step(noise_pred, t, latent)["prev_sample"]

            # decoded = self.decode_latents(latent)
            # import matplotlib.pyplot as plt
            # plt.imshow(decoded[0].cpu().permute(1,2,0).clamp(0,1))
            # plt.title("One step at" + str(i))
            # plt.show()

        return latent

    @torch.no_grad()
    def invert(self, image, prompt, cfg, num_inference_steps=25, timesteps=None):
        """

        Invert the image using the prompt, cfg for the current diffusion model using DDIM inversion
        https://github.com/inbarhub/DDPM_inversion/blob/main/ddm_inversion/ddim_inversion.py

        Args:
        - image: Float[Tensor, "B 3 512 512"]
        - prompt: str
        - cfg: float

        """

        batch_size = image.shape[0]

        text_embeddings = self.get_text_embeddings(prompt, batch_size)

        latent = self.encode_images(image)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps if timesteps is None else timesteps

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False, desc="Inverting"):

            t = timesteps[len(timesteps) - i - 1]
            t_b1 = t.repeat(batch_size).to(image.device)  # [batch_size]
            t_b2 = t.repeat(batch_size * 2).to(image.device)  # [batch_size * 2] - for unet

            latent_input = torch.cat([latent] * 2, dim=0)

            noise_pred = self.forward_unet(
                latent_input,
                t_b2,
                encoder_hidden_states=text_embeddings,
            )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

            latent = next_step_ddim_invert(self.scheduler, noise_pred, t, latent)

        return latent


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
