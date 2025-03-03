import torch

from PIL import Image
from abc import ABC, abstractmethod


class ImageInpainter(ABC):
    """
    ImageGenerator is an abstract class that defines the interface for all image generators
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def tiled_forward(
        self, img: torch.Tensor, mask: torch.Tensor, prompt: str
    ) -> torch.Tensor:
        """Inpaint the image in tiles"""
        pass

    @abstractmethod
    def __call__(
        self,
        img: torch.Tensor,  # B 3 H W in range [0, 1],
        mask: torch.Tensor,  # B 1 H W
        prompt: str,
    ) -> torch.Tensor:  # B 3 H W
        """Inpaint the image"""
        pass
