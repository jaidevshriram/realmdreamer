import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from rich import print
import json

from tqdm.auto import tqdm

from agents import setup_agent
from depth_estimators import setup_depth
from depth_estimators.zoe import ZoeDepth
from depth_estimators.depth_anything import DepthAnything
from image_generators import setup_imagen
from inpainters import setup_inpainter
from loggers import setup_logger
from pcd_handler import setup_renderer

import sys

sys.path.append("/mnt/data/temp/Portal")
from utils.seed_everything import seed_everything
from utils.image import load_image_to_pt
from utils.bilateral_filter import bilateral_filter
from utils.mask import get_outpainting_mask


# Accepts an image tensor and a function to outpaint from the image given a mask
def outpaint_from_init(prompt, img_tensor: torch.Tensor, outpaint_fn):  # B C H W

    # HARDCODED

    # Create a new image tensor with 256 pixels added to left and right sides and 128 pixels up and below
    lr_pad = 256
    ud_pad = 0
    img_tensor = F.pad(
        img_tensor, (lr_pad, lr_pad, ud_pad, ud_pad), mode="constant", value=0
    )

    # Create a mask for the new image
    mask_tensor = torch.ones(
        (img_tensor.shape[0], 1, img_tensor.shape[2], img_tensor.shape[3])
    )

    if ud_pad > 0:
        mask_tensor[:, :, ud_pad:-ud_pad, lr_pad:-lr_pad] = 0
    else:
        mask_tensor[:, :, :, lr_pad:-lr_pad] = 0

    # Inpaint the new image
    img_tensor = outpaint_fn(img=img_tensor, mask=mask_tensor, prompt=prompt)

    return img_tensor


class Generator:

    def __init__(self, cfg):

        self.cfg = cfg
        self.prompt = cfg.aux_prompt
        self.start_image = None

        # self.image_generator = setup_imagen(self.cfg.image_generator)
        self.inpainter = setup_inpainter(self.cfg.inpainter)
        self.depth_estimator = setup_depth(self.cfg.depth_estimator)
        self.renderer = setup_renderer(self.cfg.pcd_renderer)
        # self.logger = setup_logger(self.cfg.logger)
        self.agent = setup_agent(self.cfg.agent)

        self.visited_poses = []

        seed_everything(self.cfg.seed)

        print("Generator Initialized")

    def start_generation(self, img_path):

        start_image = load_image_to_pt(img_path)
        self.start_image = start_image

        # Outpaint to obtain a larger image optionally
        if self.cfg.outpaint:
            start_image = outpaint_from_init(
                self.prompt, start_image, self.inpainter.tiled_forward
            )

        init_pose = np.eye(4)

        # Estimate depth of this image and optionally give the user the chance to tweak it
        init_depth = self.depth_estimator(start_image)
        if self.cfg.depth_estimator.alignment_model == "zoedepth":
            init_depth_metric = ZoeDepth.scale_by_zoe(
                start_image,
                init_depth,
                alignment_strategy=self.cfg.depth_estimator.alignment,
            )
        elif self.cfg.depth_estimator.alignment_model == "depth_anything":
            init_depth_metric = DepthAnything.scale_by_depth_anything(
                start_image,
                init_depth,
                alignment_strategy=self.cfg.depth_estimator.alignment,
            )

        init_depth_metric = bilateral_filter(
            img=start_image, depth=init_depth_metric, depth_threshold=0.01
        )

        self.renderer.init_pcd_from_rgbd(
            rgb=start_image, depth=init_depth_metric, pose=init_pose
        )
        self.renderer.init_shadow_volume(pose=init_pose)

        self.visited_poses.append(init_pose)

        # scale, shift = self.setup_scale_and_shift(init_pointcloud)

        # init_depth = shift + scale * init_depth

        # Iterate through all poses that the agent explores

        print("Done Generating!")

    def export_to_dataset(self):

        self.agent.reset_iterator()

        nerfstudio_json = {
            "camera_model": "OPENCV",
            "fl_x": self.cfg.img_size / 2,
            "fl_y": self.cfg.img_size / 2,
            "cx": self.cfg.img_size / 2,
            "cy": self.cfg.img_size / 2,
            "w": self.cfg.img_size,
            "h": self.cfg.img_size,
            "frames": [],
        }

        # Create the directory for this scene
        output_path = os.path.join(self.cfg.output_path, self.cfg.scene_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Save the poses used
        pose_path = os.path.join(output_path, "poses.npy")
        np.save(pose_path, self.agent.poses_list)

        # Save the pointcloud
        output_ply_path = os.path.join(output_path, "pointcloud.ply")
        self.renderer.write_to_ply(output_path=output_ply_path)

        poses = []
        for i, pose in tqdm(enumerate(self.visited_poses)):

            rgb, depth, ids = self.renderer(pose, with_shadow_volume=False)
            rgb_w_shadow, depth_w_shadow, ids_w_shadow = self.renderer(
                pose, with_shadow_volume=True
            )

            mask_empty = depth < 0
            mask_object_in_shadow = ~mask_empty & (
                depth_w_shadow != depth
            )  # where the mask empty is false (cause there is an object) but it is also in shadow (depth w shadow != depth)

            mask = mask_object_in_shadow

            rgb_path = os.path.join(output_path, "rgb", f"{i}.png")
            depth_path = os.path.join(output_path, "depth", f"{i}.npy")
            mask_path = os.path.join(output_path, "mask", f"{i}.png")

            # Create the directory for RGB/Depth/Mask
            for path in [rgb_path, depth_path, mask_path]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save RGB
            rgb = transforms.ToPILImage()(rgb[0])
            rgb.save(rgb_path)

            # Save Depth
            depth_np = depth[0, 0].cpu().numpy() * 1000
            np.save(depth_path, depth_np)

            # Save Mask as PNG (invert to save)
            mask_np = 1 - mask[0, 0].cpu().float().numpy()
            mask = transforms.ToPILImage()(mask_np).convert("L")
            mask.save(mask_path)

            # # Convert to opengl - this is just for exporting back to nerfstudio - not for rendering
            transform = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )

            pose_gl = transform @ pose
            pose_gl_c2w = np.linalg.inv(pose_gl)

            # Add frame to nerfstudio json
            nerfstudio_json["frames"].append(
                {
                    "file_path": os.path.join("rgb", f"{i}.png"),
                    "depth_file_path": os.path.join("depth", f"{i}.npy"),
                    "mask_path": os.path.join("mask", f"{i}.png"),
                    "mask_inpainting_file_path": os.path.join("mask", f"{i}.png"),
                    "transform_matrix": pose_gl_c2w.tolist(),
                }
            )

        # Save nerfstudio json
        with open(os.path.join(output_path, "init_transforms.json"), "w") as f:
            json.dump(nerfstudio_json, f, indent=4)

        # Save the original image used to start the generation
        init_img_path = os.path.join(output_path, "input_rgb", "init_img.png")
        if not os.path.exists(os.path.dirname(init_img_path)):
            os.makedirs(os.path.dirname(init_img_path), exist_ok=True)

        init_img = transforms.ToPILImage()(self.start_image[0])
        init_img.convert("RGB").resize((512, 512)).save(init_img_path)

        first_img_path = os.path.join(output_path, "rgb", "0.png")
        if not os.path.exists(first_img_path):
            os.makedirs(os.path.dirname(first_img_path), exist_ok=True)
        init_img.convert("RGB").resize((512, 512)).save(first_img_path)

        print("[bold green]Done Exporting![/bold green]")


if __name__ == "__main__":

    from configs import get_cfg_defaults

    scene_names = [
        "astronaut2",
        "bedroom2",
        "bedroom3",
        "boat",
        "kong",
        "octopus",
        "resolute",
        "surf",
        # 'bedroom3', 'boat', 'kong','octopus', 'resolute', 'surf'
    ]

    depth_models = ["zoedepth", "depth_anything"]

    alignment_technique = [
        "least_squares",
        "least_squares_filtered",
        "max",
        "max_filtered",
    ]

    for scene_name in scene_names:
        for depth_model in depth_models:
            for alignment in alignment_technique:

                cfg = get_cfg_defaults()
                cfg.merge_from_file(f"configs/{scene_name}.yaml")

                cfg.scene_name = (
                    "debug_" + scene_name + "_" + depth_model + "_" + alignment
                )
                cfg.output_path = "debug_outputs"
                cfg.depth_estimator.alignment_model = depth_model
                cfg.depth_estimator.alignment = alignment

                if os.path.exists(os.path.join(cfg.output_path, cfg.scene_name)):
                    print(f"Skipping {cfg.scene_name}")
                    continue

                generator = Generator(cfg)
                generator.start_generation(cfg.base_img_path)

                if os.path.exists(os.path.join(cfg.output_path, cfg.scene_name)):
                    print(f"Skipping {cfg.scene_name}")
                    continue

                generator.export_to_dataset()

                del generator
                torch.cuda.empty_cache()

                print(
                    "Done with", cfg.scene_name + ", " + depth_model + ", " + alignment
                )
                print(50 * "-")
