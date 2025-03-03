import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

from pytorch3d.renderer import (
    AbsorptionOnlyRaymarcher,
    AlphaCompositor,
    EmissionAbsorptionRaymarcher,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    GridRaysampler,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
    RayBundle,
)
from pytorch3d.renderer import VolumeRenderer as Pytorch3dVolumeRenderer
from pytorch3d.renderer import look_at_view_transform, ray_bundle_to_ray_points

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.utils import (
    cameras_from_opencv_projection,
    opencv_from_cameras_projection,
)
from pytorch3d.io import IO

from .base import PointCloudRenderer

from utils.pcd import pytorch3d_to_o3d
from utils.occ_grid import (
    compute_grid_parameters,
    point_cloud_to_occupancy_grid,
    find_occluded_voxels,
)


# This is the regular PointsRenderer but also returns the depth map
class PointsRendererCustom(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        """
        Fragments looks something like
        
        # Class to store the outputs of point rasterization
        class PointFragments(NamedTuple):
            idx: torch.Tensor
            zbuf: torch.Tensor
            dists: torch.Tensor
        
        """

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        rgb = images[..., :3]

        return rgb, fragments.zbuf, fragments.idx


class Pytorch3DRenderer(PointCloudRenderer):
    """
    Pytorch3DRenderer is a class that renders point clouds using Pytorch3D
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.background_color = tuple([0 for _ in range(3)])
        self.shadow_volume_pcd = None

        if not torch.cuda.is_available():
            print(
                "[red]CUDA not available - required for Pytorch3D PointCloud Renderer. Exiting."
            )
            exit(1)

    @torch.no_grad()
    def rgbd_to_pcd(
        self,
        rgb: torch.Tensor,  # B C H W,
        depth: torch.Tensor,  # B 1 H W,
        pose: np.ndarray,  # 4 x 4
        mask: torch.Tensor = None,  # B 1 H W,
    ):

        assert rgb.shape[0] == 1, "Only support a batch size of 1"
        assert (
            len(rgb.shape) == 4 and len(depth.shape) == 4 and len(pose.shape) == 2
        ), "Input sizes incorrect"

        if mask is not None:
            assert len(mask.shape) == 4, "Mask shape incorrect"

        img = rgb.permute(0, 2, 3, 1)[0]  # Now it is H, W, C
        depth = depth[0]  # Now it is 1 H W
        camera = self.np_pose_to_cam(pose).to("cuda")

        top_y, top_x = (-1, -1)

        ndc_coords_y = torch.linspace(-top_y, top_y, img.shape[1], device="cuda")
        ndc_coords_x = torch.linspace(-top_x, top_x, img.shape[0], device="cuda")
        Y, X = torch.meshgrid(ndc_coords_x, ndc_coords_y)

        xy_depth = torch.dstack([X, Y, depth.squeeze(0)]).to("cuda")  # H W
        points = camera.unproject_points(
            xy_depth,
            in_ndc=False,
            from_ndc=False,
            world_coordinates=True,
        )

        if mask is not None:
            mask = mask[0]
            points = points[mask.squeeze(0) > 0.5]
            img = img[mask.squeeze(0) > 0.5]

        points = points.reshape(-1, 3)
        img = img.reshape(-1, 3)

        pcd = Pointclouds(points=[points], features=[img])

        return pcd

    def init_pcd_from_rgbd(
        self,
        rgb: torch.Tensor,  # B C H W,
        depth: torch.Tensor,  # B 1 H W,
        pose: np.ndarray,  # 4 4
    ):

        assert rgb.shape[0] == 1, "Only supports a batch size of 1"

        rgb = rgb.to("cuda")
        depth = depth.to("cuda")

        self.pcd = self.rgbd_to_pcd(rgb, depth, pose)

    def init_shadow_volume(self, pose: np.ndarray, voxel_size=0.05):  # 4 4

        # Convert current point cloud to open3d point cloud
        pcd_o3d = pytorch3d_to_o3d(self.pcd)

        # Get grid dimensions of point cloud
        min_corner, dims = compute_grid_parameters(pcd_o3d, voxel_size)

        total_size = np.prod(dims)
        max_total_size = 100000000

        if total_size > max_total_size:

            print(
                f"Total size of grid is {total_size}. This is too large for shadow volume calculation."
            )

            multiplier = (max_total_size / total_size) ** (1 / 3)

            assert multiplier < 1, "Multiplier should be less than 1"

            voxel_size = voxel_size / multiplier

            min_corner, dims = compute_grid_parameters(pcd_o3d, voxel_size)

            print("Multiplier : ", multiplier)
            print("New voxel size : ", voxel_size)

            print(f"Using voxel size of {voxel_size} and dims {dims}")
            print(f"Total number of voxels: {np.prod(dims)}")

        # Convert PCD to occupancy grid
        occ_grid = point_cloud_to_occupancy_grid(pcd_o3d, voxel_size, min_corner, dims)

        # Find occluded voxels
        pose_pos = pose[:3, 3]
        occluded_voxels = find_occluded_voxels(
            occ_grid, pose_pos, min_corner, voxel_size
        )

        # Create a new point cloud at the voxel centers - set to shadow volume PCD
        voxel_coords = np.argwhere(occluded_voxels > 0)
        voxel_centers = voxel_coords * voxel_size + min_corner

        points = torch.from_numpy(voxel_centers).to("cuda").float()
        colors = (
            torch.from_numpy(np.ones((voxel_centers.shape[0], 3)) * 0.5)
            .float()
            .to("cuda")
        )

        self.shadow_volume_pcd = Pointclouds(points=[points], features=[colors])

    def update_pcd(
        self,
        rgb,  # B C H W,
        depth,  # B 1 H W,
        mask,  # B 1 H W,
        pose,  # B 4x4 matrix
    ):

        assert rgb.shape[0] == 1, "Only supports a batch size of 1"

        new_pcd = self.rgbd_to_pcd(rgb, depth, pose, mask)
        self.pcd = self.combine_pcds(self.pcd, new_pcd)

    @torch.no_grad()
    def __call__(
        self, pose: np.ndarray, img_size=512, with_shadow_volume=True
    ):  # pose is a 4x4 matrix

        if with_shadow_volume:
            assert (
                self.shadow_volume_pcd is not None
            ), "Shadow volume PCD is not initialized"
            pcd = self.combine_pcds(self.pcd, self.shadow_volume_pcd)
        else:
            pcd = self.pcd

        cameras = self.np_pose_to_cam(pose).to("cuda")

        radius = 0.005
        points_per_pixel = 1

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to rasterize_points.py for explanations of these parameters.
        raster_settings = PointsRasterizationSettings(
            image_size=(img_size, img_size),
            radius=radius,
            points_per_pixel=points_per_pixel,
            bin_size=0,
        )

        # Create a points renderer by compositing points using an weighted compositor (3D points are
        # weighted according to their distance to a pixel and accumulated using a weighted sum)
        renderer = PointsRendererCustom(
            rasterizer=PointsRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            # compositor=AlphaCompositor(background_color=self.background_color)
            compositor=NormWeightedCompositor(
                background_color=self.background_color
            ),  # The fourth channel is the uncertainity
        )

        images, depth, ids = renderer(pcd)

        return (
            images.permute(0, 3, 1, 2),
            depth[:, :, :, 0].unsqueeze(1),
            ids[:, :, :, 0].unsqueeze(1),
        )  # img is B, 3, H, W # Only get the closest depth

    @staticmethod
    def combine_pcds(point_cloud1: Pointclouds, point_cloud2: Pointclouds):
        """
        Combines two point clouds into one
        """

        # If one of the point clouds is None, return the other
        if point_cloud1 is None:
            return point_cloud2

        if point_cloud2 is None:
            return point_cloud1

        # Get the points and colors of the two point clouds
        points1 = point_cloud1.points_padded()
        colors1 = point_cloud1.features_padded()

        points2 = point_cloud2.points_padded()
        colors2 = point_cloud2.features_padded()

        if colors1 is None:
            return point_cloud2

        if colors2 is None:
            return point_cloud1

        # Combine the points and colors
        points = torch.cat([points1, points2], dim=1).squeeze(0)
        colors = torch.cat([colors1, colors2], dim=1).squeeze(0)

        # Create a new point cloud
        point_cloud = Pointclouds(points=[points], features=[colors])

        return point_cloud

    def delete_points(self, ids):
        """
        Deletes points from the point cloud
        """

        # Get the points and colors of the point cloud
        points = self.pcd.points_padded()
        colors = self.pcd.features_padded()

        # Get the points and colors that are not in the ids
        mask = torch.ones(points.shape[1], dtype=bool).to("cuda")

        # set the mask to False for the ids
        mask[ids] = False

        # Apply the mask
        points = points[:, mask]
        colors = colors[:, mask]

        # Create a new point cloud
        self.pcd = Pointclouds(points=[points], features=[colors])

    @staticmethod
    def np_pose_to_cam(pose, f=256, img_size=512):
        """
        Convert a pose from OpenCV format to pytorch3d FOV format
        Input:
            pose: 4x4 matrix
        Output:
            cameras: PerspectiveCameras object
        """

        # Convert to pytorch3d format
        R = torch.from_numpy(pose[:3, :3]).float()
        T = torch.from_numpy(pose[:3, 3]).float()
        T = T.reshape(1, 3)
        R = R.reshape(1, 3, 3)

        K = torch.eye(3)[None, ...].float()
        K[..., 0, 0] = K[..., 1, 1] = f
        K[..., 0, 2] = K[..., 1, 2] = f

        img_size = torch.ones(1, 2) * img_size

        # cameras = PerspectiveCameras(R=R, T=T, focal_length=[img_size], principal_point=[(img_size/2, img_size/2)], in_ndc=False, image_size=[(img_size, img_size)])
        cameras = cameras_from_opencv_projection(
            R=R, tvec=T, camera_matrix=K, image_size=img_size
        )

        return cameras

    def write_to_ply(self, output_path: str):
        """
        Export the pointcloud to a PLY
        """

        pcd_points = self.pcd.points_padded()
        pcd_colors = self.pcd.features_padded() * 255

        pcd = Pointclouds(points=pcd_points, features=pcd_colors)

        IO().save_pointcloud(pcd, output_path)

    def load_from_ply(self, input_path: str):
        """
        Load a pointcloud from a PLY
        """

        pcd = o3d.io.read_point_cloud(input_path)
        pcd_points = torch.from_numpy(np.asarray(pcd.points)).float().unsqueeze(0)
        pcd_colors = torch.from_numpy(np.asarray(pcd.colors)).float().unsqueeze(0)

        self.pcd = Pointclouds(points=pcd_points, features=pcd_colors).to("cuda")
