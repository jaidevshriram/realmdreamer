from .sd_2 import StableDiffusion2

# from .sdxl import StableDiffusionXL


def setup_imagen(cfg):
    """Sets up an image generator"""

    if cfg.model == "stable-diffusion-2":
        return StableDiffusion2(cfg)
    elif cfg.model == "stable-diffusion-xl":
        return StableDiffusionXL(cfg)
    else:
        raise NotImplementedError(cfg.model)
