import numpy as np
import torch
import pdb
import base64

from PIL import Image

from .base import DepthEstimator

from .extern.GeoWizard.geowizard.models.geowizard_pipeline import (
    DepthNormalEstimationPipeline,
)

from utils.depth import align_depths, blend_depths

from utils.bilateral_filter import bilateral_filter
from torchvision.utils import save_image
from litellm import completion


import kornia


def pt_to_pil(img: torch.Tensor) -> Image:
    img = img.cpu().numpy()
    img = img.transpose(1, 2, 0)

    img = Image.fromarray(np.uint8(img * 255))
    return img


class GeoWizard(DepthEstimator):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model = DepthNormalEstimationPipeline.from_pretrained(
            "lemonaddie/geowizard"
        ).to("cuda")

        if torch.cuda.is_available():
            self.model.to("cuda")

        self.mode = "unknown"

        if cfg.mode in ["indoor", "outdoor"]:
            self.mode = cfg.mode

    def overlap_depth_pred(
        self,
        img: torch.Tensor,  # B C H W
        depth_gt: torch.Tensor,  # B 1 H W
        mask_new: torch.Tensor,  # B 1 H W
    ):

        # Align the two depths
        mask_valid_depth = depth_gt > 0
        depth_pred_unmasked = self(img)
        depth_pred_aligned_unmasked = align_depths(
            source_depth=depth_pred_unmasked,
            target_depth=depth_gt,
            mask=mask_valid_depth,
        )
        depth_pred_aligned_unmasked[mask_valid_depth] = depth_gt[mask_valid_depth]

        inpaint_mask = ~mask_valid_depth
        inpaint_mask = kornia.morphology.dilation(
            inpaint_mask.float(), kernel=torch.ones((10, 10), device=img.device)
        ).bool()

        depth_pred = self.inpaint(
            img, depth_pred_aligned_unmasked, inpaint_mask
        )  # Inpaint the non valid part of depth

        # plt.imshow(depth_pred[0, 0].cpu().numpy())
        # plt.colorbar()
        # plt.show()

        depth_pred_aligned = align_depths(
            source_depth=depth_pred, target_depth=depth_gt, mask=mask_valid_depth
        )

        # Blend the depths at the edges
        # depth_blended = blend_depths(source_depth=depth_pred_aligned, target_depth=depth_gt, mask_new=inpaint_mask)
        depth_blended = depth_pred_aligned

        return depth_blended

    def inpaint(
        self, img: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:

        b, c, h, w = img.shape

        outputs = []
        for i in range(b):
            img_pil = pt_to_pil(img[i])
            output = self.model(
                img_pil,
                ensemble_size=10,
                denoising_steps=10,
                processing_res=512,
                domain=self.mode,
                show_progress_bar=True,
            )
            output_pt = torch.from_numpy(output.depth_np)
            outputs.append(output_pt)

        outputs = torch.stack(outputs, dim=0)

        return outputs.unsqueeze(1).to(img.device)

    def wide_img_inference(
        self,
        img: torch.Tensor,  # B C H W
    ) -> torch.Tensor:
        """
        Predicts depth for an image that may be wider than 512x512
        """

        raise NotImplementedError

        import matplotlib.pyplot as plt

        b, c, h, w = img.shape

        # go through tiles of length 512 in steps of 256
        skip_size = 256

        # Number of tiles = ceil(height / 256) x ceil(width / 256)
        height_tiles = int(np.ceil(h / skip_size))
        width_tiles = int(np.ceil(w / skip_size))

        mask = torch.ones((b, 1, h, w))
        outputs = torch.ones((b, 1, h, w))

        for i in range(b):

            for tile_hi in range(height_tiles):
                for tile_wi in range(width_tiles):

                    # If a tile goes beyond the edge, shift it back
                    tile_height_start = skip_size * tile_hi
                    tile_height_end = tile_height_start + 512

                    if tile_height_end >= h:
                        diff = np.abs(h - tile_height_end)
                        tile_height_end -= diff
                        tile_height_start -= diff

                    tile_width_start = skip_size * tile_wi
                    tile_width_end = tile_width_start + 512

                    if tile_width_end >= w:
                        diff = np.abs(w - tile_width_end)
                        tile_width_start -= diff
                        tile_width_end -= diff

                    rgb_in = img[
                        i,
                        :,
                        tile_height_start:tile_height_end,
                        tile_width_start:tile_width_end,
                    ].unsqueeze(0)
                    depth_in = outputs[
                        i,
                        :,
                        tile_height_start:tile_height_end,
                        tile_width_start:tile_width_end,
                    ].unsqueeze(0)
                    mask_in = mask[
                        i,
                        :,
                        tile_height_start:tile_height_end,
                        tile_width_start:tile_width_end,
                    ].unsqueeze(0)

                    tile_output = self.model.inpaint(
                        rgb_in=rgb_in, depth_in=depth_in, mask=mask_in
                    ).depth_np

                    # print(outputs[i, :, tile_height_start:tile_height_end, tile_width_start:tile_width_end].shape)
                    outputs[
                        i,
                        :,
                        tile_height_start:tile_height_end,
                        tile_width_start:tile_width_end,
                    ] = torch.from_numpy(tile_output).unsqueeze(0)

                    mask[
                        i,
                        :,
                        tile_height_start:tile_height_end,
                        tile_width_start:tile_width_end,
                    ] = 0

        return outputs

    def __call__(self, img: torch.Tensor) -> torch.Tensor:  # B C H W

        # Pass image to GPT 4 and determine if it is indoor or outdoor
        if self.mode == "unknown":
            # Save image to a temporary path
            tmp_path = "/tmp/base_img.jpg"
            save_image(img, tmp_path)

            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

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
                    "GPT 4 did not return a valid scene type while using GeoWizard"
                )

            self.mode = scene_type

        b, c, h, w = img.shape

        outputs = []
        for i in range(b):
            img_pil = pt_to_pil(img[i])
            output = self.model(
                img_pil,
                ensemble_size=10,
                denoising_steps=10,
                processing_res=512,
                domain=self.mode,
                show_progress_bar=True,
            )
            output_pt = torch.from_numpy(output.depth_np)
            outputs.append(output_pt)

        outputs = torch.stack(outputs, dim=0)

        return outputs.unsqueeze(1).to(img.device)
