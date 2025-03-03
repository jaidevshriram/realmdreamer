import numpy as np
import torch

from .gs_camera import Camera as GaussianSplattingCamera
from .gs_graphics_utils import focal2fov, fov2focal, getWorld2View2


def ns2gs_camera(ns_camera, device=None):

    batch_size = ns_camera.camera_to_worlds.shape[0]

    if device is None:
        device = ns_camera.camera_to_worlds.device

    assert batch_size > 0, "Batch size must be greater than 0"

    if batch_size > 1 and len(ns_camera.camera_to_worlds.shape) > 2:

        cameras = []

        for i in range(batch_size):

            c2w = torch.clone(ns_camera.camera_to_worlds[i, 0, ...])
            c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w.cpu().numpy())
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            FovY = focal2fov(ns_camera.fy[i, 0, 0], ns_camera.height[i, 0, 0])
            FovX = focal2fov(ns_camera.fx[i, 0, 0], ns_camera.width[i, 0, 0])

            cameras.append(
                GaussianSplattingCamera(
                    R=R,
                    T=T,
                    width=ns_camera.width[i, 0, 0],
                    height=ns_camera.height[i, 0, 0],
                    FoVx=FovX,
                    FoVy=FovY,
                    data_device=device,
                )
            )

        return cameras

    else:

        if len(ns_camera.camera_to_worlds.shape) > 2:
            c2w = torch.clone(ns_camera.camera_to_worlds[0, 0, ...])
        else:
            c2w = torch.clone(ns_camera.camera_to_worlds)

        c2w = c2w.to(device)
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w.cpu().numpy())
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(ns_camera.fy, ns_camera.height)
        FovX = focal2fov(ns_camera.fx, ns_camera.width)

        return GaussianSplattingCamera(
            R=R,
            T=T,
            width=ns_camera.width,
            height=ns_camera.height,
            FoVx=FovX,
            FoVy=FovY,
            data_device=device,
        )
