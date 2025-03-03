from .pytorch3d import Pytorch3DRenderer
from .open3d import Open3DRenderer


def setup_renderer(cfg):
    if cfg.name == "pytorch3d":
        return Pytorch3DRenderer(cfg)
    elif cfg.name == "open3d":
        return Open3DRenderer(cfg)
    else:
        raise NotImplementedError(f"Unknown renderer: {cfg.name}")
