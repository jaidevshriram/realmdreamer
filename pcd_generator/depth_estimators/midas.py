import torch

from .base import DepthEstimator


class Midas(DepthEstimator):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=True
        )

        if torch.cuda.is_available():
            self.model.to("cuda")

        self.model.eval()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        disparity = self.model(img)

        depth = 1000 / disparity

        return depth
