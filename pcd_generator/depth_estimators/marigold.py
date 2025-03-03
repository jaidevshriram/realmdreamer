import numpy as np
import torch
import pdb

from PIL import Image
from .extern.marigold.marigold_pipeline import MarigoldPipeline

from .base import DepthEstimator

from utils.depth import align_depths, blend_depths

from utils.bilateral_filter import bilateral_filter

import kornia


def pt_to_pil(img: torch.Tensor) -> Image:
    img = img.cpu().numpy()
    img = img.transpose(1, 2, 0)

    img = Image.fromarray(np.uint8(img * 255))
    return img


class Marigold(DepthEstimator):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model = MarigoldPipeline.from_pretrained(
            "Bingxin/Marigold", torch_dtype=torch.float16
        )

        if torch.cuda.is_available():
            self.model.to("cuda")

    def overlap_depth_pred(
        self,
        img: torch.Tensor,  # B C H W
        depth_gt: torch.Tensor,  # B 1 H W
        mask_new: torch.Tensor,  # B 1 H W
    ):

        import matplotlib.pyplot as plt

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

        plt.imshow(depth_pred[0, 0].cpu().numpy())
        plt.colorbar()
        plt.show()

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
            output = self.model.inpaint(
                rgb_in=img, depth_in=depth, mask=mask, num_inference_steps=10
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

        b, c, h, w = img.shape

        outputs = []
        for i in range(b):
            img_pil = pt_to_pil(img[i])
            output = self.model(img_pil, ensemble_size=10, denoising_steps=10)
            output_pt = torch.from_numpy(output.depth_np)
            outputs.append(output_pt)

        outputs = torch.stack(outputs, dim=0)

        return outputs.unsqueeze(1).to(img.device)
