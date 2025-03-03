import copy
import math
import pdb
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Type

import numpy as np
import open3d as o3d
import torch
from nerfstudio.fields.base_field import (Field, FieldConfig,
                                          get_normalized_directions)
from nerfstudio.utils.rich_utils import CONSOLE
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import Tensor, nn

from .gs_sh_utils import *
from .gs_utils import (build_rotation, build_scaling_rotation,
                       get_expon_lr_func, inverse_sigmoid,
                       inverse_wide_sigmoid, strip_symmetric)
from .pcd_utils import (K_nearest_neighbors, compute_grid_parameters,
                        distance_to_gaussian_surface, find_occluded_voxels,
                        point_cloud_to_occupancy_grid)
from .transforms import qsvec2rotmat_batched, qvec2rotmat_batched


@dataclass
class GaussianSplattingFieldConfig(FieldConfig):

    _target: Type = field(default_factory=lambda: GaussianSplattingField)

    max_sh_degree: int = 0
    """The maximum degree of the spherical harmonics"""

    percent_dense: float = 0.1
    """Percent dense of the scene extent"""


class GaussianSplattingField(Field):
    def __init__(
        self,
        config: GaussianSplattingFieldConfig,
    ) -> None:

        super().__init__()

        self.config = config

        self.active_sh_degree = 0
        self.max_sh_degree = config.max_sh_degree

        self._xyz = torch.empty(0)  # XYZ coordinates

        self._features_dc = torch.empty(0)  # The point cloud features (such as colour)
        self._features_rest = torch.empty(0)

        self._scaling = torch.empty(0)  # The scaling of the splats
        self._rotation = torch.empty(0)  # The rotation of the splats
        self._opacity = torch.empty(0)  # The opacity of the splats
        # self.max_radii2D = torch.empty(0) # The maximum radii of the splats in screen space

        self.xyz_gradient_accum = torch.empty(0)  # The accumulated gradient for each point
        self.xyz_gradient_accum_list = []  # The accumulated gradient for each point
        self.denom = torch.empty(0)  # The number of times each point has been updated

        self.percent_dense = config.percent_dense

        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def load_pcd(
        self,
        path,
        occluded_rand_init=False,
        init_requires_grad=True,
        auto_scale=True,
        device="cuda",
        use_sigmoid=True,
    ):
        """
        Read the point cloud from a PLY file and initialise the gaussian splat.

        occluded_rand_init: bool - Initializes the shadow volume if set
        init_requires_grad: bool - Whether gradients should flow to points set at initialization
        auto_scale: bool - Set scale of gaussians based on proximity to nearest points or a fixed constant
        """

        self.device = device

        import open3d as o3d

        print("Loading PCD file : ", path)

        if type(path) == str:
            pcd = o3d.io.read_point_cloud(path)
        else:
            pcd = path

        self.base_pcd = copy.deepcopy(pcd)
        self.num_points_init = None
        points = np.asarray(pcd.points)
        num_points = points.shape[0]

        CONSOLE.print("Number of points in OG point cloud : ", points.shape[0])

        if occluded_rand_init:

            print("Computing occluded voxels...")

            voxel_size = 0.05
            min_corner, dims = compute_grid_parameters(pcd, voxel_size)

            total_size = np.prod(dims)
            max_total_size = 1000000

            if total_size > max_total_size:
                print(f"Total size of grid is {total_size}. This is too large for shadow volume calculation.")

                multiplier = (max_total_size / total_size) ** (1 / 3)

                assert multiplier < 1, "Multiplier should be less than 1"

                voxel_size = voxel_size / multiplier

                min_corner, dims = compute_grid_parameters(pcd, voxel_size)

                print("Multiplier : ", multiplier)
                print("New voxel size : ", voxel_size)

                print(f"Using voxel size of {voxel_size} and dims {dims}")
                print(f"Total number of voxels: {np.prod(dims)}")

            viewpoint = np.array([0.0, 0.0, 0.0])
            self.occ_grid = point_cloud_to_occupancy_grid(pcd, voxel_size, min_corner, dims)

            print("Occupancy grid created with dims : ", dims)

            start = time.time()
            occluded_voxels = find_occluded_voxels(self.occ_grid, viewpoint, min_corner, voxel_size)
            end = time.time()

            print("Time taken to find occluded voxels : ", end - start)
            # print("Occluded voxel content", occluded_voxels.min(), occluded_voxels.max())

            voxel_coords = np.argwhere(occluded_voxels > 0)
            voxel_centers = voxel_coords * voxel_size + min_corner

            print("Number of occluded voxels : ", voxel_centers.shape[0])

            # Create a new point cloud at the voxel centers
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(voxel_centers)

            # Gray color for the newpoints
            pcd_new.colors = o3d.utility.Vector3dVector(np.ones((voxel_centers.shape[0], 3)) / 2)

            # Combine the two points clouds
            pcd = pcd + pcd_new

        points = np.asarray(pcd.points)

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(device)

        colors = torch.tensor(np.asarray(pcd.colors)).float().to(self.device)

        if use_sigmoid:
            colors = inverse_wide_sigmoid(colors)

        fused_color = RGB2SH(colors)

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)

        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        CONSOLE.print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)),
            0.0000001,
        )

        if auto_scale:
            print("Autoscaling points...")

            # Autoscale only the initial points
            dist2_init_points = torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)[: self.num_points_init, :]),
                0.0000001,
            )

            # scale in the initial region
            scales = torch.zeros_like(dist2)[..., None].repeat(1, 3)
            scales[: self.num_points_init, :] = torch.log(torch.sqrt(dist2_init_points))[..., None].repeat(1, 3)

            # Do not scale points in the occluded region
            scales[self.num_points_init :, :] = (
                torch.ones_like(dist2[self.num_points_init :])[..., None].repeat(1, 3) * -3.5951
            )
        else:
            print("Scale set v low")
            scales = torch.ones_like(dist2)[..., None].repeat(1, 3) * -10.5951

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device)

        # Store the initial number of points used
        if self.num_points_init is None:
            del self.num_points_init
            self.register_buffer("num_points_init", torch.tensor(num_points, device=self.device))

        # Save the positions of the initial points
        self.init_xyz = fused_point_cloud[: self.num_points_init]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(init_requires_grad))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(init_requires_grad)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(init_requires_grad)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(init_requires_grad))
        self._rotation = nn.Parameter(rots.requires_grad_(init_requires_grad))
        self._opacity = nn.Parameter(opacities.requires_grad_(init_requires_grad))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        self.active_sh_degree = self.max_sh_degree

        self.add_retain_hooks()

    # Clear the gradient for the num_points_init rows of parameters using hook
    def add_retain_hooks(self):
        return

        # def retain_grad_hook(grad):

        #     try:
        #         grad = grad.clone()
        #         grad[:self.num_points_init] = 0.0
        #         return grad
        #     except:
        #         return grad

        # self._xyz.register_hook(retain_grad_hook)
        # self._features_dc.register_hook(retain_grad_hook)
        # self._features_rest.register_hook(retain_grad_hook)
        # self._scaling.register_hook(retain_grad_hook)
        # self._rotation.register_hook(retain_grad_hook)
        # self._opacity.register_hook(retain_grad_hook)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def get_averaged_features(self, num_neighbours=5):
        features_dc = self._features_dc
        features_rest = self._features_rest

        features = torch.cat((features_dc, features_rest), dim=1)  # N x 1 x 3 (3 if only rgb)

        # Get the indices of the nearest points
        _, idx = K_nearest_neighbors(
            self._xyz, K=num_neighbours + 1, return_dist=False, include_original=True
        )  # N x num_neighbours+1

        neighbour_colors = features[idx.detach(), :, :]  # N x num_neighbours x 1 x 3

        averaged_colors = torch.sum(neighbour_colors, dim=1) / num_neighbours

        return averaged_colors

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest

        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_rotmat(self):
        return qvec2rotmat_batched(self.get_rotation)

    @property
    def get_covariance(self):
        return qsvec2covmat_batched(self.get_rotation, self.get_scaling)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def add_densification_stats(self, viewspace_point_tensors, update_filters):

        bs = len(viewspace_point_tensors)
        viewspace_point_tensor_grads = [viewspace_point_tensors[i].grad for i in range(bs)]

        # Combine the gradients from all the elements of the batch
        for i in range(bs):

            self.xyz_gradient_accum[update_filters[i]] += torch.norm(
                viewspace_point_tensor_grads[i][update_filters[i], :2],
                dim=-1,
                keepdim=True,
            )
            self.denom[update_filters[i]] += 1

    def cat_tensors_to_optimizer(self, tensors_dict, optimizers):
        optimizable_tensors = {}
        for param_name in tensors_dict.keys():

            group = optimizers[param_name].param_groups[0]

            assert len(group["params"]) == 1, f"Assert failed {param_name}"
            extension_tensor = tensors_dict[param_name]
            stored_state = optimizers[param_name].state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del optimizers[param_name].state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizers[param_name].state[group["params"][0]] = stored_state

                optimizable_tensors[param_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        optimizers,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d, optimizers)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.xyz_gradient_accum_list = [[] for i in range(self.get_xyz.shape[0])]
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def reload_optimizer(self, optimizers):

        mask = torch.zeros_like(self.get_opacity).squeeze().bool()
        self.prune_points(mask, optimizers)

    def _prune_optimizer(self, mask, optimizers):
        optimizable_tensors = {}
        for param_name in optimizers.keys():

            if param_name in ["guidance"]:
                continue

            group = optimizers[param_name].param_groups[0]
            stored_state = optimizers[param_name].state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizers[param_name].state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizers[param_name].state[group["params"][0]] = stored_state

                optimizable_tensors[param_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[param_name] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, optimizers):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizers)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]

    def break_big_splats(self, optimizers, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition

        selected_pts_mask = self.scaling_inverse_activation(torch.max(self.get_scaling, dim=1).values) > -2

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            optimizers,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
            )
        )
        self.prune_points(prune_filter, optimizers)

        return selected_pts_mask.sum().item()

    def densify_and_split(self, grads, grad_threshold, scene_extent, optimizers, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            optimizers,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
            )
        )
        self.prune_points(prune_filter, optimizers)

        return selected_pts_mask.sum().item()

    def densify_and_split_until_threshold(self, grads, grad_threshold, scene_extent, optimizers, N=2):

        total = 0
        counter = 0
        while True or counter < 5:

            n_init_points = self.get_xyz.shape[0]

            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device=self.device)
            padded_grad[: grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
            )

            if selected_pts_mask.sum().item() == 0:
                break

            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                optimizers,
            )

            prune_filter = torch.cat(
                (
                    selected_pts_mask,
                    torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
                )
            )
            self.prune_points(prune_filter, optimizers)

            counter += 1
            total += selected_pts_mask.sum().item()

        return total

    def densify_and_clone(self, grads, grad_threshold, scene_extent, optimizers):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            optimizers,
        )

        return selected_pts_mask.sum().item()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, optimizers):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        split_new = self.densify_and_split(grads, max_grad, extent, optimizers)

        torch.cuda.empty_cache()

        self.add_retain_hooks()

        return {
            "new_points": split_new,
            # 'compact_new': compact_new,
            "max_grad": grads.max(),
        }

    def prune(self, min_opacity, extent, max_screen_size, optimizers):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 9999999.0

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        self.prune_points(prune_mask, optimizers)

        torch.cuda.empty_cache()

        self.add_retain_hooks()

        return {
            "pruned_points": prune_mask.sum().item(),
        }

    def replace_tensor_to_optimizer(self, tensor, param_name, optimizers):
        optimizable_tensors = {}

        group = optimizers[param_name].param_groups[0]

        assert len(group["params"]) == 1, f"Assert failed {param_name}"
        stored_state = optimizers[param_name].state.get(group["params"][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del optimizers[param_name].state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizers[param_name].state[group["params"][0]] = stored_state

            optimizable_tensors[param_name] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    def reset_opacity(self, optimizers):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity", optimizers)
        self._opacity = optimizable_tensors["opacity"]

    def densify_by_compatnes_with_idx(self, idx, mark=False, filter_scale=False):
        nn_svec = self.get_scaling[idx]
        nn_rotmat = self.get_rotmat[idx]
        nn_pos = self._xyz[idx]

        # Compute the distance from the nearest neighbours of a gaussian to the original gaussian's surface
        nn_gaussian_surface_dist = distance_to_gaussian_surface(nn_pos, nn_svec, nn_rotmat, self._xyz)

        # Compute the distance from the original gaussian to the nearest neighbours of a gaussian's surface
        gaussian_surface_dist = distance_to_gaussian_surface(self._xyz, self.get_scaling, self.get_rotmat, nn_pos)

        # The distance to the nearest neighbour (euclidean distance)
        dist_to_nn = torch.norm(nn_pos - self._xyz, dim=-1)

        # We want to densify in places where the distance to the nearest neighbour is more than the sum of the radius of the two gaussians in the direction of the other
        mask = (gaussian_surface_dist + nn_gaussian_surface_dist) < dist_to_nn

        # We want the new mean to be in the middle of the two gaussians
        new_direction = (nn_pos - self._xyz.data) / dist_to_nn[..., None]
        new_mean = (
            self._xyz.data
            + new_direction * (dist_to_nn + gaussian_surface_dist - nn_gaussian_surface_dist)[..., None] / 2.0
        )[mask]

        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)

        # set a random direction to render colours
        dir_pp = torch.randn_like(self._xyz.data)
        dir_pp_normalized = dir_pp / torch.norm(dir_pp, dim=-1, keepdim=True)

        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        new_raw_color = colors_precomp[mask]  # rgb

        if mark:

            raise Warning("Why mark the points - are you sure?")

            new_raw_color[:, 0] = 1.0
            new_raw_color[:, 1] = 0.412
            new_raw_color[:, 2] = 0.706
        else:
            pass

        new_raw_color = RGB2SH(new_raw_color)  # SH

        new_qvec = self._rotation.data[mask]

        new_raw_alpha = self._opacity.data[mask]

        new_raw_svec = self.scaling_inverse_activation(
            torch.ones_like(self.get_scaling.data[mask])
            * (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask][..., None]
            / 6.0
        )

        # Remove all the points that are too small
        if filter_scale:
            point_mask = torch.max(new_raw_svec, dim=1).values > -4.605
        else:
            point_mask = torch.ones_like(new_raw_svec[:, 0]) == 1

        new_mean = new_mean[point_mask]
        new_raw_color = new_raw_color[point_mask]
        new_qvec = new_qvec[point_mask]
        new_raw_alpha = new_raw_alpha[point_mask]
        new_raw_svec = new_raw_svec[point_mask]

        new_params = {
            "xyz": new_mean,
            "rotation": new_qvec,
            "scaling": new_raw_svec,
            "f_dc": new_raw_color.unsqueeze(1),
            "f_rest": torch.zeros_like(self._features_rest.data[mask])[point_mask],
            "opacity": new_raw_alpha,
        }

        return new_params

    def densify_by_compactness(self, K, optimizers, mark=False, filter_scale=True):

        start = time.time()
        _, idx = K_nearest_neighbors(self._xyz, K=K + 1, return_dist=False)
        end = time.time()

        # print("Nearest neighbour time", end - start)

        return 0

        num_densified = 0
        new_params_list = []

        for i in range(K):
            new_params = self.densify_by_compatnes_with_idx(idx[:, i], mark=mark, filter_scale=filter_scale)
            new_params_list.append(new_params)

        new_params = {}
        for key in new_params_list[0].keys():
            new_params[key] = torch.cat([p[key] for p in new_params_list], dim=0)
        num_densified = new_params["xyz"].shape[0]

        new_xyz = new_params["xyz"]
        new_features_dc = new_params["f_dc"]
        new_features_rest = new_params["f_rest"]
        new_opacities = new_params["opacity"]
        new_scaling = new_params["scaling"]
        new_rotation = new_params["rotation"]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            optimizers,
        )

        return num_densified

    # Functions that we don't care too much for
    def init_random(self, num_points=1000):

        points = 1 * (torch.rand((num_points, 3), device="cuda") - 0.5)
        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.rand((num_points, 3, 1), device="cuda").transpose(1, 2).requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.rand((num_points, 3, (self.max_sh_degree + 1) ** 2 - 1), device="cuda")
            .transpose(1, 2)
            .requires_grad_(True)
        )
        # self._scaling = nn.Parameter(torch.rand((num_points, 3), device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.ones((num_points, 3), device="cuda").requires_grad_(True) * -4.5951)

        u = torch.rand(num_points, 1, device="cuda")
        v = torch.rand(num_points, 1, device="cuda")
        w = torch.rand(num_points, 1, device="cuda")

        # self._rotation = torch.cat(
        #     [
        #         torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        #         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        #         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        #         torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        #     ],
        #     -1,
        # )
        self._rotation = nn.Parameter(torch.randn((num_points, 4), device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.ones((num_points, 1), device="cuda").requires_grad_(True))
        # self.max_radii2D = torch.zeros((num_points), device="cuda")
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
        self.xyz_gradient_accum_list = [[] for i in range(num_points)]
        self.denom = torch.zeros((num_points, 1), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def init_random_from_pcd(self, path, occluded_rand_init=False, init_requires_grad=True):

        import open3d as o3d

        print("Loading PCD file : ", path)

        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)

        CONSOLE.print("Number of points in OG point cloud : ", points.shape[0])

        points = np.asarray(pcd.points)
        points = np.asarray(pcd.points) + np.random.randn(points.shape[0], 3) * 0.1

        colors = np.asarray(pcd.colors)
        colors = np.random.rand(colors.shape[0], 3)

        fused_point_cloud = torch.tensor(points).float().cuda()
        fused_color = RGB2SH(torch.tensor(colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        CONSOLE.print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(points).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Keep track of the number of points used
        self.num_points_init = points.shape[0]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(init_requires_grad))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(init_requires_grad)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(init_requires_grad)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(init_requires_grad))
        self._rotation = nn.Parameter(rots.requires_grad_(init_requires_grad))
        self._opacity = nn.Parameter(opacities.requires_grad_(init_requires_grad))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def backproject_and_combine(self, rgb, depth, mask, pose):
        """
        This function backprojects the RGBD image into a point cloud and combines it with the current point cloud

        rgb: torch.Tensor of shape (3, H, W)
        depth: torch.Tensor of shape (1, H, W)
        mask: torch.Tensor of shape (1, H, W)
        pose: torch.Tensor of shape (4, 4)
        """

        # set depth values outside mask to be very high (will get ignored )
        depth[mask != 1] = torch.nan

        rgb = rgb.clamp(0, 1)
        rgbi = o3d.geometry.Image((rgb.cpu().contiguous().numpy() * 255).astype(np.uint8))
        depthi = o3d.geometry.Image(depth.cpu().numpy())

        # Create a RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgbi,
            depthi,
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False,
        )

        # to do: not hardcode the intrinsics
        # Create a 90 degree FOV intrinsics matrix
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_matrix.set_intrinsics(512, 512, 256.0, 256.0, 256.0, 256.0)

        if pose.shape[0] == 3 and pose.shape[1] == 4:
            pose = np.vstack((pose, np.array([0, 0, 0, 1])))

        # pose is w2c in opengl convention
        pose = np.linalg.inv(pose)

        # flip y and z axis
        pose[1, :] = -pose[1, :]
        pose[2, :] = -pose[2, :]

        # Create a point cloud from the RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic_matrix,
            pose,
        )

        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors) / 255)

        # Combine the base point cloud with the new point cloud
        self.load_pcd(self.base_pcd + pcd)

        return pcd

    def load_from_ckpt(self, path):

        ckpt = torch.load(path, map_location="cuda")

        for key in ckpt["pipeline"].keys():
            if "gaussian_model" in key:
                new_key = key.split(".")[-1]
                ckpt[new_key] = ckpt["pipeline"][key]

        self._xyz = nn.Parameter(ckpt["_xyz"].requires_grad_(True))
        self._features_dc = nn.Parameter(ckpt["_features_dc"].contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(ckpt["_features_rest"].contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(ckpt["_scaling"].requires_grad_(True))
        self._rotation = nn.Parameter(ckpt["_rotation"].requires_grad_(True))
        self._opacity = nn.Parameter(ckpt["_opacity"].requires_grad_(True))

        print(ckpt["_opacity"].min(), ckpt["_opacity"].max())

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.active_sh_degree = self.max_sh_degree
