import open3d as o3d
import numpy as np
from tqdm.auto import tqdm
import pdb
import torch
import time

import torch.nn.functional as F
from typing import Optional

from multiprocessing import Pool, cpu_count

from utils.line import Bresenham3D

import occlude

def compute_grid_parameters(pc, voxel_size):
    bounding_box = pc.get_axis_aligned_bounding_box()
    min_corner = np.asarray(bounding_box.min_bound)
    max_corner = np.asarray(bounding_box.max_bound)

    dims = np.ceil((max_corner - min_corner) / voxel_size).astype(int)
    return min_corner, dims


def point_cloud_to_occupancy_grid(pc, voxel_size, min_corner, dims):
    grid = np.zeros(dims, dtype=np.uint8)
    for point in pc.points:
        index = np.floor((point - min_corner) / voxel_size).astype(int)
        if np.all(index >= 0) and np.all(index < dims):
            grid[tuple(index)] = 1
        else:
            print("Point outside grid: ", point)

    return grid


def find_occluded_voxels(grid, viewpoint, min_corner, voxel_size):

    #     # grid - N x N x N - np.ndarray
    #     # viewpoint - 3 - np.ndarray
    #     # origin - 3 - np.ndarray
    #     # voxel_size - float

    origin_grid = np.floor((viewpoint - min_corner) / voxel_size).astype(int)

    # print(grid.shape, origin_grid.shape, origin_grid, grid.min(), grid.max())
    # np.save("/mnt/data/Portal/outputs/castle_combined/grid.npy", grid)

    visible_voxels = occlude.find_occluded_voxels(
        grid,
        list(origin_grid),
        (0, 0, 0),
        1,
        grid.shape[0],
        grid.shape[1],
        grid.shape[2],
    )

    # print("Computation done?")
    visible_voxels = np.array(visible_voxels)

    return (visible_voxels < 1).astype(int)
