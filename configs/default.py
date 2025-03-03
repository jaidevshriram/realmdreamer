# Default Configuration values and structure of Configuration Node

from yacs.config import CfgNode as CN

_C = CN()

# Project level configuration
_C.project_name = "Portal"

_C.scene_name = "living_room"  # No spaces
_C.seed = 215
_C.base_img_path = "data/images/castle.png"
_C.device = "cuda"
_C.img_size = 512
_C.debug = True
_C.prompt = "A living room"
_C.aux_prompt = ""
_C.import_scene = False
_C.import_scene_path = ""
_C.outpaint = False

# Output paths
_C.output_path = "outputs"

"""Agent Options:
- dummy - will load pose paths
"""
_C.agent = CN()
_C.agent.name = "dummy"
_C.agent.poses_path = "data/poses/look_left_right_back_left_right.npy"

"""
Inpainter Options:
- stable-diffusion-2-inpainting
"""
_C.inpainter = CN()
_C.inpainter.name = "stable-diffusion-2-inpainting"
_C.inpainter.num_images = (
    4  # The number of images to generate before choosing the best one
)

"""
Depth Estimator Options:
- midas
- marigold
"""
_C.depth_estimator = CN()
_C.depth_estimator.name = "geowizard"
_C.depth_estimator.alignment_model = "depth_anything"  # "zoedepth" or "depth_anything"
_C.depth_estimator.alignment = "least_squares_filtered"  # "least_squares" or "least_squares_filtered" or "max" or "max_filtered"
_C.depth_estimator.mode = ""  # "indoor" or "outdoor"

"""
Point Cloud Renderer Options
- pytorch3d
"""
_C.pcd_renderer = CN()
_C.pcd_renderer.name = "pytorch3d"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
