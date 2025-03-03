from __future__ import annotations

import json
import math
import os
import pdb
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Tuple, Type, Union

import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from jaxtyping import Bool, Float
from kornia.color.colormap import AUTUMN
from kornia.filters import gaussian_blur2d, laplacian
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import (
    DepthLossType,
    L1Loss,
    MSELoss,
    ScaleAndShiftInvariantLoss,
    depth_loss,
    distortion_loss,
    interlevel_loss,
    lossfun_outer,
    orientation_loss,
    pred_normal_loss,
    ray_samples_to_sdist,
    scale_gradients_by_distance_squared,
    tv_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import ViewerSlider
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.functional.regression import pearson_corrcoef

from realmdreamer.gaussian_splatting.gs_camera import Camera as GaussianSplattingCamera
from realmdreamer.gaussian_splatting.gs_camera_utils import ns2gs_camera
from realmdreamer.gaussian_splatting.gs_field import (
    GaussianSplattingField,
    GaussianSplattingFieldConfig,
)
from realmdreamer.gaussian_splatting.gs_graphics_utils import (
    focal2fov,
    fov2focal,
    getWorld2View2,
)
from realmdreamer.gaussian_splatting.gs_sh_utils import eval_sh
from realmdreamer.gaussian_splatting.gs_utils import wide_sigmoid


@dataclass
class GaussianSplattingPCDModel(ModelConfig):
    _target: Type = field(default_factory=lambda: GaussianSplatting)

    gaussian_model: GaussianSplattingFieldConfig = GaussianSplattingFieldConfig()
    """Config for the Gaussian model."""

    background_color: str = "white"
    """Background color for the rendered images. Either "black" or "white"."""

    sh_degree: int = 0
    """Maximum SH degree to use for rendering."""

    pcd_path: str = None
    """Path to the model to load."""


class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianSplatting(Model):
    config: GaussianSplattingPCDModel
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        device,
        load_iteration: int = -1,
        orientation_transform: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(config, scene_box, num_train_data)

        self.config = config

        self.gaussian_model.to("cuda")
        self.gaussian_model.xyz_gradient_accum.to("cuda")
        self.gaussian_model.denom.to("cuda")

        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

    def populate_modules(self):
        super().populate_modules()

        # load gaussian model
        self.gaussian_model = self.config.gaussian_model.setup()

        self.gaussian_model.load_pcd(
            os.path.join(self.config.pcd_path),
            occluded_rand_init=False,
            device="cuda",
            use_sigmoid=False,
        )
        # self.gaussian_model.init_random_from_pcd(os.path.join(self.config.pcd_path), occluded_rand_init=False)

        self.gaussian_model.xyz_gradient_accum = torch.zeros(
            (self.gaussian_model.get_xyz.shape[0], 1), device="cuda"
        )
        self.gaussian_model.denom = torch.zeros(
            (self.gaussian_model.get_xyz.shape[0], 1), device="cuda"
        )

    def get_loss_dict(self, outputs, batch, prompt, c2w, step_ratio):
        return {}

    @torch.cuda.amp.autocast(False)
    def forward(self, cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a camera pose
        Args:
            - c2w: camera poses
        """

        viewpoint_cameras = ns2gs_camera(cameras, device="cuda")

        # assert not isinstance(viewpoint_cameras, list), "Cameras are a list is a list"
        if isinstance(viewpoint_cameras, list):

            outputs = {
                "rgb": [],
                "depth": [],
                "alpha": [],
                "viewspace_points": [],
                "visibility_filter": [],
                "radii": [],
            }

            for viewpoint_camera in viewpoint_cameras:
                out = self.forward_single(viewpoint_camera)
                for key in out.keys():
                    outputs[key].append(out[key])

            for key in outputs.keys():
                if key != "viewspace_points":
                    outputs[key] = torch.stack(outputs[key], dim=0)
        else:

            outputs = self.forward_single(viewpoint_cameras)

            for key in outputs.keys():
                if key == "viewspace_points":
                    outputs[key] = [outputs[key]]
                else:
                    outputs[key] = outputs[key].unsqueeze(0)

        return outputs

    def forward_single(self, viewpoint_camera) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Run forward for a single camera pose
        """

        background = torch.tensor(
            self.bg_color, dtype=torch.float32, device=self.device
        )

        # random background
        # random_background = torch.tensor(np.random.uniform(0, 1, (3,)), dtype=torch.float32, device=self.device)

        start = time.time()
        out = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
            # bg_color=random_background
        )
        end = time.time()

        # print(out['render'].shape, end - start, "IMAGE")

        image = out["render"]
        depth = out["depth"]
        alpha = out["alpha"]
        viewspace_points = out["viewspace_points"]
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]

        return {
            "rgb": image,
            "depth": depth,
            "alpha": alpha,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle
    ) -> Dict[str, torch.Tensor]:
        viewpoint_camera = ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(
            self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device
        )

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
            # opacity_scale=opacity_scale,
            # scale_modifier=scale_modifier,
        )

        render = render_results["render"]
        depth = render_results["depth"]
        alpha = render_results["alpha"]

        rgb = torch.permute(torch.clamp(render, max=1.0), (1, 2, 0))
        depth = torch.permute(depth, (1, 2, 0))
        return {
            "rgb": rgb,
            "depth": depth,
        }

    def render(
        self,
        viewpoint_camera,
        pc,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussian_model.get_xyz,
                dtype=self.gaussian_model.get_xyz.dtype,
                requires_grad=True,
                device=pc.get_xyz.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(self.device)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )

        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        # print(means3D.device, means3D.device, colors_precomp.device, opacity.device, scales.device, rotations.device)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        visibility_filter = radii > 0

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        return param_groups

    def named_parameters(self, recurse=False):
        params = {}
        return params

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        return metrics_dict

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
        pass
