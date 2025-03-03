import pdb
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from tqdm.auto import tqdm

import occlude

from .line_utils import Bresenham3D


# adapted from gsgen
def distance_to_gaussian_surface(mean, svec, rotmat, query):

    # mean - N x 3 - torch.Tensor (mean position)
    # svec - N x 3 - torch.Tensor (scale vector)
    # rotmat - N x 3 x 3 - torch.Tensor (rotation matrix)
    # query - N x 3 - torch.Tensor (query points)

    xyz = query - mean
    # TODO: check here
    # breakpoint()

    xyz = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), xyz)
    xyz = F.normalize(xyz, dim=-1)

    z = xyz[..., 2]
    y = xyz[..., 1]
    x = xyz[..., 0]

    r_xy = torch.sqrt(x**2 + y**2 + 1e-10)

    cos_theta = z
    sin_theta = r_xy
    cos_phi = x / r_xy
    sin_phi = y / r_xy

    d2 = svec[..., 0] ** 2 * cos_phi**2 + svec[..., 1] ** 2 * sin_phi**2
    r2 = svec[..., 2] ** 2 * cos_theta**2 + d2**2 * sin_theta**2

    return torch.sqrt(r2 + 1e-10)


# adapted from gsgen
@torch.no_grad()
def K_nearest_neighbors(
    mean: torch.Tensor,
    K: int,
    query: Optional[torch.Tensor] = None,
    return_dist=False,
    include_original=False,
):

    # TODO: finish this
    if query is None:
        query = mean
    dist, idx, nn = knn_points(query[None, ...], mean[None, ...], K=K, return_nn=True)

    if include_original:
        start = 0
    else:
        start = 1

    if not return_dist:
        return nn[0, :, start:, :], idx[0, :, start:]
    else:
        return nn[0, :, start:, :], idx[0, :, start:], dist[0, :, start:]


def compute_grid_parameters(pc, voxel_size):
    bounding_box = pc.get_axis_aligned_bounding_box()
    min_corner = np.asarray(bounding_box.min_bound)
    max_corner = np.asarray(bounding_box.max_bound)

    dims = np.ceil((max_corner - min_corner) / voxel_size).astype(int)
    return min_corner, dims


def point_cloud_to_occupancy_grid(pc, voxel_size, min_corner, dims):
    grid = np.zeros(dims, dtype=bool)
    for point in pc.points:
        index = np.floor((point - min_corner) / voxel_size).astype(int)
        if np.all(index >= 0) and np.all(index < dims):
            grid[tuple(index)] = True
        else:
            print("Point outside grid: ", point)

    return grid


def occupancy_grid_to_point_cloud(occ_grid, min_corner, voxel_size):

    pc = o3d.geometry.PointCloud()
    points = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if occ_grid[i, j, k] == 1:
                    points.append(min_corner + np.array([i, j, k]) * voxel_size)

    pc.points = o3d.utility.Vector3dVector(np.asarray(points))
    pc.colors = o3d.utility.Vector3dVector(np.randn((len(points), 3)))
    return pc


def find_occluded_voxels(grid, viewpoint, min_corner, voxel_size):
    """
    Find occluded voxels in a 3D grid from a given viewpoint

    Args:
     # grid - N x N x N - np.ndarray
     # viewpoint - 3 - np.ndarray
     # origin - 3 - np.ndarray
     # voxel_size - float
    """

    origin_grid = np.floor((viewpoint - min_corner) / voxel_size).astype(int)

    visible_voxels = occlude.find_occluded_voxels(
        grid,
        list(origin_grid),
        (0, 0, 0),
        1,
        grid.shape[0],
        grid.shape[1],
        grid.shape[2],
    )

    print("Occluded volume computed")

    visible_voxels = np.array(visible_voxels)

    return (visible_voxels < 1).astype(int)
