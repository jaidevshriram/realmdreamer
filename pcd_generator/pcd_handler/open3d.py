import open3d as o3d
import numpy as np
import torch

from .base import PointCloudRenderer


class Open3DRenderer(PointCloudRenderer):
    """
    PointCloudRenderer is an abstract class that defines point cloud renderers
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.points = None
        self.colors = None
        self.pcd = o3d.geometry.PointCloud()

    def rgbd_to_pcd(
        self,
        rgb: torch.Tensor,  # B C H W
        depth: torch.Tensor,  # B 1 H W
        pose: np.ndarray,  # 4x4
        mask: torch.Tensor = None,  # B 1 H W
    ):

        assert rgb.shape[0] == 1, "Support only batch size of 1"
        assert (
            rgb.shape[-1] == 512 and rgb.shape[-2] == 512
        ), "Supports only img size of 512x512"

        rgb = rgb[0].permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = np.uint8(rgb * 255)

        depth = depth[0].permute(1, 2, 0).cpu().contiguous().numpy().squeeze()
        # depth = np.uint8(depth)

        print(rgb.shape, rgb.dtype, rgb.min(), rgb.max(), depth.dtype)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth)
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width=512, height=512, fx=256, fy=256, cx=256, cy=256
            ),
        )

        pcd.transform(pose)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.points)

        if mask is not None:
            mask = mask.squeeze().flatten().cpu().numpy()

            points = points[mask]
            colors = colors[mask]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud

    def init_pcd_from_rgbd(
        self,
        rgb: torch.Tensor,  # B C H W,
        depth: torch.Tensor,  # B 1 H W,
        pose: np.ndarray,  # 4x4
    ):

        pcd = self.rgbd_to_pcd(rgb=rgb, depth=depth, pose=pose)
        self.pcd = pcd

    def __call__(self, pose, with_shadow_volume=False):

        if with_shadow_volume:
            raise NotImplementedError("Shadow volume not implemented for open3d")

        # Create a renderer without opening a window
        renderer = o3d.visualization.draw_geometries([self.pcd])

        # Set camera intrinsics and extrinsics
        # renderer.scene.camera.intrinsic = intrinsics
        renderer.scene.camera.extrinsic = pose

        # Capture RGB image
        rgb_image = renderer.capture_screen_float_buffer(False)

        # Capture depth image
        depth_image = renderer.capture_depth_float_buffer(False)

        # Convert depth image to a numpy array
        depth_data = np.asarray(depth_image)

        # Convert RGB image to a numpy array
        rgb_data = np.asarray(rgb_image)

        rgb_t = torch.from_numpy(rgb_data)
        depth_t = torch.from_numpy(depth_data)

        return rgb_t, depth_t

    def update_pcd(
        self,
        rgb: torch.Tensor,  # B C H W,
        depth: torch.Tensor,  # B 1 H W,
        mask: torch.Tensor,  # B 1 H W,
        pose: np.ndarray,  # 4x4 matrix
    ):
        """Converts the image, depth and mask to a point cloud and updates the existing point cloud."""

        assert rgb.shape[0] == 1, "Only supports a batch size of 1"

        new_pcd = self.rgbd_to_pcd(rgb, depth, pose, mask)
        self.pcd = self.combine_pcds(self.pcd, new_pcd)

    def __call__(self, pose: np.ndarray):  # pose is a 4x4 matrix
        return self.render(pose)

    @staticmethod
    def combine_pcds(pcd1, pcd2):

        points1 = np.asarray(pcd1.points)
        colors1 = np.asarray(pcd1.colors)

        points2 = np.asarray(pcd2.points)
        colors2 = np.asarray(pcd2.colors)

        combined_points = np.concatenate((points1, points2), axis=0)
        combined_colors = np.concatenate((colors1, colors2), axis=0)

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

        return combined_pcd
