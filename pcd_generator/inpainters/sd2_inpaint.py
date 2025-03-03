import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import StableDiffusionInpaintPipeline

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
)

from sentence_transformers import SentenceTransformer, util

from .base import ImageInpainter


class StableDiffusion2Inpainter(ImageInpainter):

    def __init__(self, cfg):

        super().__init__(cfg)

        pipe_kwargs = {
            "requires_safety_checker": False,
        }

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            **pipe_kwargs,
        )

        self.inference_steps = 50

        if torch.cuda.is_available():
            # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
            # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            self.inference_steps = 25

            pipe = pipe.to("cuda")

        self.model = pipe
        self.pipe = pipe
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.weights_dtype = torch.float16
        self.num_train_timesteps = 1000
        self.scheduler.set_timesteps(self.num_train_timesteps)

        self.pil_to_tensor = transforms.ToTensor()
        self.num_images = cfg.num_images

        # Load the CLIP model
        self.clip_model = SentenceTransformer("clip-ViT-B-32")

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
    def decode_latents(
        self,
        latents,  # Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        #  -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

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

    def get_text_embeddings(self, prompt, batch_size):
        """
        Get text embeddings for a given prompt - per batch size
        """

        text_embeddings = self.pipe._encode_prompt(
            prompt=prompt,
            negative_prompt="ugly, text, chairs, tables, furniture, poor lighting, lowres, low quality",
            device="cuda",
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # Split into cond and uncond
        text_embeddings_uncond, text_embeddings_cond = text_embeddings.chunk(2)

        # Combine it back into cond and unconcd
        text_embeddings = torch.cat(
            [text_embeddings_cond, text_embeddings_uncond], dim=0
        )

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
            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )

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

            timesteps = torch.round(
                torch.arange(init_timestep, 0, -init_timestep / num_inference_steps)
            ).to(scheduler.timesteps.dtype)

            return timesteps, num_inference_steps, init_timestep

    @torch.no_grad()
    def __call__(
        self, img: torch.Tensor, mask: torch.Tensor, prompt: str
    ) -> torch.Tensor:
        """Inpaint the image"""

        # If mask is empty, return the original image
        if torch.sum(mask) == 0:
            return img

        # Map the image to the range of -1 to 1
        img = img * 2 - 1

        negative_prompt = (
            "ugly, text, furniture, chairs, tables, bed, numbers, dark, shutterstock"
        )
        self.model.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.model.scheduler.config
        )
        inpainted = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            mask_image=mask,
            guidance_scale=5,
            num_inference_steps=20,
            num_images_per_prompt=self.num_images,
        ).images

        # Choose the image with the highest similarity
        inpainted = self.img_picker(inpainted, prompt)

        return self.pil_to_tensor(inpainted).unsqueeze(0)

    def tiled_forward(
        self, img: torch.Tensor, mask: torch.Tensor, prompt: str
    ) -> torch.Tensor:
        """Will run the inpainting model in tiles over the whole image - for resolutions larger than 512x512"""

        img = img * 2 - 1

        b, c, h, w = img.shape
        tile_size = 512

        overlap = 100
        stride = tile_size - overlap

        for i in range(0, h, stride):
            for j in range(0, w, stride):

                start_x = j
                start_y = i

                end_x = min(start_x + tile_size, w)
                end_y = min(start_y + tile_size, h)

                # Adjust the starting coordinates if the remaining portion is not a square
                if end_x - start_x < tile_size:
                    start_x = max(end_x - tile_size, 0)
                if end_y - start_y < tile_size:
                    start_y = max(end_y - tile_size, 0)

                img_tile = img[:, :, start_y:end_y, start_x:end_x]
                # img_tile_pil = Image.fromarray(np.uint8(img_tile.numpy()))

                # Mask for tile - all pixels which are black
                mask_tile = mask[:, :, start_y:end_y, start_x:end_x]

                imgs_tile = self.model(
                    prompt=prompt,
                    image=img_tile,
                    mask_image=mask_tile,
                    num_inference_steps=self.inference_steps,
                    num_images_per_prompt=1,
                ).images

                img_tile = self.img_picker(imgs_tile, prompt)
                # img_tile = imgs_tile[0]

                img_tile_tensor = (
                    torch.from_numpy(np.array(img_tile))
                    .to(img.device)
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                )

                # print(img_tile_tensor.shape, img_tile_tensor.min(), img_tile_tensor.max())

                img[:, :, start_y:end_y, start_x:end_x] = (
                    img_tile_tensor / 255
                ) * 2 - 1
                mask[:, :, start_y:end_y, start_x:end_x] = 0

        return (img + 1) / 2

    def img_picker(self, imgs: torch.Tensor, prompt: str):
        """Choose the most aligned image semantically from a set of images w.r.t a prompt"""

        # Extract the CLIP embeddings of the generated images
        clip_embedding_imgs = self.clip_model.encode(imgs)

        clip_embedding_prompt = self.clip_model.encode(prompt)

        # Calculate the cosine similarity between the CLIP embeddings of the generated images and the prompt
        similarities = util.cos_sim(clip_embedding_imgs, clip_embedding_prompt)

        # Choose the image with the highest similarity
        img = imgs[np.argmax(similarities)]

        return img
