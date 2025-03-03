from .midas import Midas
from .marigold import Marigold
from .geowizard import GeoWizard


def setup_depth(cfg):
    if cfg.name == "midas":
        return Midas(cfg)
    elif cfg.name == "marigold":
        return Marigold(cfg)
    elif cfg.name == "geowizard":
        return GeoWizard(cfg)
    else:
        raise NotImplementedError(f"Unknown depth estimator: {cfg.name}")
