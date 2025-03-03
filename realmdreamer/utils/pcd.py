import numpy as np
import open3d as o3d
from pytorch3d.structures import Pointclouds


def pytorch3d_to_o3d(pcd_pytorch3d: Pointclouds):

    points = pcd_pytorch3d.points_padded().squeeze().cpu()
    colors = pcd_pytorch3d.features_padded().squeeze().cpu()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud
