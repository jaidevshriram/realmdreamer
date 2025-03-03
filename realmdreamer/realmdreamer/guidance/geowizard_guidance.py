# https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_guidance.py

import pdb
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cprint import *
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from torch import Tensor
from torchmetrics.functional.regression import pearson_corrcoef
from tqdm.auto import tqdm

from realmdreamer.extern.GeoWizard.geowizard.models.geowizard_pipeline import \
    DepthNormalEstimationPipeline


def align_tensors_by_median(tensor1, tensor2):

    tensor1_median = torch.median(tensor1)
    tensor2_median = torch.median(tensor2)

    return tensor1, tensor2 * (tensor1_median / (tensor2_median + 1e-5))


# Alex
def get_scale_translation(
    x: np.ndarray,  # (N,)
    y: np.ndarray,  # (N,)
):
    """
    Given two N dimensions tensors, compute a solution to a * x + b = y. Return a, b
    """

    x = x[:, None]  # (N, 1)
    x = np.concatenate([x, np.ones(shape=x.shape)], -1)  # (N, 2)
    psuedo = np.linalg.inv(x.T @ x) @ x.T

    scale, translation = (psuedo @ y[..., None]).squeeze()

    return scale, translation


def align_depths(
    source_depth: torch.Tensor,  # B 1 H W
    target_depth: torch.Tensor,  # B 1 H W
    mask: torch.Tensor = None,  # B 1 H W
    enforce_scale_positive=False,
):
    """
    Given two depth maps, align one with the other. If a mask is provided, will use only the values within the mask
    """

    assert (
        source_depth.shape == target_depth.shape
    ), f"Shape of input depths is not the same {source_depth.shape} vs {target_depth.shape}"

    assert mask.shape == source_depth.shape, f"Shape of mask is not same as depth {source_depth.shape} vs {mask.shape}"

    batch_size, _, _, _ = source_depth.shape

    output_depth = source_depth.clone()

    for i in range(batch_size):

        source_depth_i = source_depth[i].squeeze()  # H W
        target_depth_i = target_depth[i].squeeze()

        if mask is not None:
            mask_i = mask[i].squeeze()
            source_depth_i = source_depth_i[mask_i]
            target_depth_i = target_depth_i[mask_i]

        scale, translation = get_scale_translation(
            source_depth_i.flatten().detach().cpu().numpy(),
            target_depth_i.flatten().detach().cpu().numpy(),
        )

        # print(scale, "is scale")
        if scale < 0:

            if enforce_scale_positive:  # Exit early and return none if scale is meant to be positive
                return None

            print("Scale is negative!!!", scale)
            scale = 1
            # raise Warning("Scale should not be negative", scale)

        output_depth[i] = translation + scale * source_depth

    return output_depth  # B 1 H W


class GeoWizardConfig:
    pretrained_model_name_or_path: str = "lemonaddie/geowizard"
    enable_channels_last_format: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    weighting_strategy: str = "sds"

    anneal: bool = False


class GeoWizardGuidance(nn.Module):

    def __init__(
        self,
        device: Union[torch.device, str],
        # full_precision=False,
        min_step_percent: float = 0.02,
        max_step_percent: float = 0.98,
    ) -> None:
        super().__init__()

        self.cfg = GeoWizardConfig()
        self.device = device

        self.cfg.min_step_percent = min_step_percent
        self.cfg.max_step_percent = max_step_percent
        self.full_precision = True

        self.weights_dtype = torch.float32

        self.pipe = DepthNormalEstimationPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
        ).to(self.device)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        # self.scheduler.alphas = self.scheduler.alphas.to(self.device)

        cprint.info(f"Loaded GeoWizard!")

    @torch.no_grad()
    def pred_one_step(self, rgb_in, depth_in=None, strength=1.0, pbar=False):  # B C H W

        depth, normal = self.pipe.single_infer(
            rgb_in,
            single_step=True,
            num_inference_steps=10,
            domain="indoor",
            show_pbar=False,
        )

        return depth


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value
