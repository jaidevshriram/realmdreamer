from .wandb_logger import WandbLoggerCustom


def setup_logger(cfg):
    return WandbLoggerCustom(cfg)
