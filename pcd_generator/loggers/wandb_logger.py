import torch
import wandb


class WandbLoggerCustom:
    """
    This is a custom logger for wandb. It is used to log images to wandb.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.wandb_logger = wandb.init(
            project=cfg.project_name,
            name=cfg.scene_name,
            config=cfg,
            # reinit=True
        )

        self.step = 0

    def log_image(self, image, name):

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            image = (image * 255).astype("uint8")

        self.wandb_logger.log({name: wandb.Image(image)}, step=self.step)

    def log_images(self, images, names):

        for image, name in zip(images, names):
            self.log_image(image, name)

    def update_step(self):
        self.step += 1
