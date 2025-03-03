import torch
from .base import DepthEstimator

import pdb


class ZoeDepth(DepthEstimator):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=True
        )

        if torch.cuda.is_available():
            self.model.to("cuda")

        self.model.eval()

    @staticmethod
    def scale_by_zoe(
        img: torch.Tensor,  # 1 C H W
        depth_unscaled: torch.Tensor,  # 1 C H W
        alignment_strategy: str = "least_squares",  # "least_squares" or "least_squares_filtered" or "max" or "max_95"
    ):
        """
        Computes the metric depth using ZoeDepth and aligns the input depth with it as best as possible.
        """

        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)

        with torch.no_grad():
            depth_metric = model(img)["metric_depth"]

        assert depth_unscaled.shape == depth_metric.shape, (
            "Input depth and metric depth must have the same shape, current"
            + str(depth_unscaled.shape)
            + " and "
            + str(depth_metric.shape)
        )

        depth_metric_aligned = DepthEstimator.align_depths(
            source_depth=depth_unscaled,
            target_depth=depth_metric,
            alignment_strategy=alignment_strategy,
        )

        return depth_metric_aligned

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        disparity = self.model(img)

        depth = 1000 / disparity

        return depth
