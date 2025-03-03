"""

This script will load a PLY file and render it using poses from a numpy file

"""

import argparse
import json
import os

import kornia
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from nerfstudio.cameras.cameras import Cameras, CameraType
from realmdreamer.dummy.gs_model_render import (GaussianSplatting,
                                                GaussianSplattingPCDModel)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init",
        type=str,
        help="Initial transform file, which will be appended to",
        required=True,
    )
    parser.add_argument("--ply", type=str, help="Path to PLY file", required=True)
    parser.add_argument(
        "--traj", type=str, help="Path to render trajectory JSON file", required=True
    )
    args = parser.parse_args()

    scene_name = args.ply.split("/")[-2].split(".")[0]

    print("Rendering scene: ", scene_name)

    # Create model
    model_config = GaussianSplattingPCDModel(pcd_path=args.ply)
    model = model_config._target(
        model_config, scene_box=None, num_train_data=1, device="cuda"
    )
    model.gaussian_model.load_pcd(
        args.ply, occluded_rand_init=False, device="cuda", use_sigmoid=False
    )
    model.to("cuda")

    # Load nerfstudio transform
    transform_file = json.load(open(args.init, "r"))

    # Load trajectory
    traj = json.load(open(args.traj, "r"))

    poses = np.array(
        [
            np.fromstring(frame["matrix"].strip("[]"), sep=",").reshape(4, 4).T
            for frame in traj["keyframes"]
        ]
    )

    # Render
    for i, pose in tqdm(enumerate(poses)):

        # Convert a pose to a camera
        camera = Cameras(
            fx=torch.tensor(256.0).cuda(),
            fy=torch.tensor(256.0).cuda(),
            cx=torch.tensor(256.0).cuda(),
            cy=torch.tensor(256.0).cuda(),
            width=torch.tensor(512).cuda(),
            height=torch.tensor(512).cuda(),
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=torch.from_numpy(pose[:3, :]).to("cuda"),
            # camera_to_worlds=torch.from_numpy(np.eye(4)[:3, :]).to('cuda')
        )

        # Render
        renders = model(camera)

        inverse_alpha = 1 - renders["alpha"]

        for _ in range(2):
            inverse_alpha = kornia.morphology.closing(
                inverse_alpha, kernel=torch.ones(5, 5).to("cuda")
            )

        for _ in range(2):
            inverse_alpha = kornia.morphology.opening(
                inverse_alpha, kernel=torch.ones(5, 5).to("cuda")
            )

        rgb = renders["rgb"].permute(0, 2, 3, 1)[0].squeeze()
        depth = renders["depth"].permute(0, 2, 3, 1)[0].squeeze()
        # alpha = renders['alpha'].permute(0, 2, 3, 1)[0].squeeze()
        inverse_alpha = inverse_alpha.permute(0, 2, 3, 1)[0].squeeze()

        # Save rgb tensor to file
        rgb = rgb.detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()
        inverse_alpha = inverse_alpha.detach().cpu().numpy()
        # alpha = alpha.detach().cpu().numpy()

        # Get inpainting mask from alpha

        rgb_path = os.path.join("outputs", scene_name, "rgb_extra", f"{i}.png")
        depth_path = os.path.join("outputs", scene_name, "depth_extra", f"{i}.npy")
        mask_path = os.path.join("outputs", scene_name, "mask_extra", f"{i}.png")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        # Save to file
        rgb = (rgb * 255).astype(np.uint8)

        Image.fromarray(rgb).convert("RGB").save(rgb_path)
        # np.save(rgb_path, rgb)
        np.save(depth_path, depth * 1000)
        Image.fromarray(((1 - inverse_alpha) * 255).astype(np.uint8)).save(mask_path)

        transform_file["frames"].append(
            {
                "file_path": os.path.join("rgb_extra", f"{i}.png"),
                "depth_file_path": os.path.join("depth_extra", f"{i}.npy"),
                "mask_path": os.path.join("mask_extra", f"{i}.png"),
                "mask_inpainting_file_path": os.path.join("mask_extra", f"{i}.png"),
                "transform_matrix": pose.tolist(),
            }
        )

    # Save the transform file
    json.dump(
        transform_file, open(f"outputs/{scene_name}/transforms.json", "w"), indent=4
    )
