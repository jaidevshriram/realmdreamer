from .sd2_inpaint import StableDiffusion2Inpainter


def setup_inpainter(cfg):
    """Setup an inpainting model"""

    if cfg.name == "stable-diffusion-2-inpainting":
        return StableDiffusion2Inpainter(cfg)
    else:
        raise NotImplementedError(cfg.name)
