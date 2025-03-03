from abc import ABC, abstractmethod

from PIL import Image
from rich.console import Console


class ImageGenerator(ABC):
    """
    ImageGenerator is an abstract class that defines the interface for all image generators
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.console = Console()

    @abstractmethod
    def __call__(self, prompt: str, height: int, width: int) -> Image:
        pass
