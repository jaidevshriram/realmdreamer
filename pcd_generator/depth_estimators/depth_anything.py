import base64
import pdb

import litellm
import torch
from depth_anything.metric_depth.zoedepth.models.builder import build_model
from depth_anything.metric_depth.zoedepth.utils.config import get_config
from litellm import completion
from torch.nn.functional import interpolate
from torchvision.utils import save_image

import sys
from .base import DepthEstimator


class DepthAnything(DepthEstimator):

    def __init__(self, cfg):

        super().__init__(cfg)

        if cfg.mode not in ["indoor", "outdoor"]:
            raise ValueError(
                "Invalid mode specified for Depth Anything, must be 'indoor' or 'outdoor'"
            )

        self.mode = cfg.mode

    @staticmethod
    def scale_by_depth_anything(
        img: torch.Tensor,  # 1 C H W
        depth_unscaled: torch.Tensor,  # 1 C H W
        alignment_strategy: str = "least_squares",  # "least_squares" or "least_squares_filtered" or "max" or "max_95"
        mode="none"
    ):
        """
        Computes the metric depth using Depth Anything and aligns the input depth with it as best as possible.
        """

        # Save image to a temporary path
        tmp_path = "/tmp/base_img.jpg"
        save_image(img, tmp_path)

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        if mode not in ["indoor", "outdoor"]:
            # Determine if it is an indoor or outdoor scene
            prompt = "You are shown the following image. Tell me if it is an outdoor or indoor scene. Output the word 'indoor' or 'outdoor' without quotes."
            encoded_image = encode_image(tmp_path)
            response = completion(
                model="gpt-4-vision-preview",
                max_tokens=5,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                },
                            },
                        ],
                    }
                ],
            )

            # Extract answer from the GPT 4 response
            scene_type = response["choices"][0]["message"]["content"]

            if scene_type not in ["indoor", "outdoor"]:
                raise ValueError(
                    "GPT 4 did not return a valid scene type while using Depth Anything"
                )
        else:
            scene_type = mode

        # Build the model
        config = get_config("zoedepth", "infer", "nyu")
        config.pretrained_resource = f"local::./pcd_generator/depth_estimators/checkpoints/depth_anything_metric_depth_{scene_type}.pt"
        model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        # Run the model
        with torch.no_grad():
            depth_metric = model(img.to(model.device))["metric_depth"]

        if depth_metric.shape[2] != depth_metric.shape[3]:
            print("The depth is not square that is fishy - check!")

        depth_metric = interpolate(
            depth_metric,
            (depth_unscaled.shape[2], depth_unscaled.shape[3]),
            mode="nearest",
        )

        assert depth_unscaled.shape == depth_metric.shape, (
            "Input depth and metric depth must have the same shape, current"
            + str(depth_unscaled.shape)
            + " and "
            + str(depth_metric.shape)
        )

        # Align the depths using the specified strategy
        if scene_type == "indoor":
            depth_metric_aligned = DepthEstimator.align_depths(
                source_depth=depth_unscaled,
                target_depth=depth_metric,
                alignment_strategy="max_filtered",
            )
        elif scene_type == "outdoor":
            depth_metric_aligned = DepthEstimator.align_depths(
                source_depth=depth_unscaled,
                target_depth=depth_metric,
                alignment_strategy="least_squares_filtered",
            )

        return depth_metric_aligned

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError(
            "Depth Anything is not yet implemented for regular use"
        )
