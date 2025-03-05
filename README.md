<h1 align="center">RealmDreamer: Text-Driven 3D Scene Generation
with Inpainting and Depth Diffusion (3DV '25)</h1>
<p align="center">
    <a href="https://realmdreamer.github.io/">[Project Page]</a> | <a href="https://arxiv.org/abs/2404.07199">[Arxiv]</a>
    <br>
    <br>
    <picture>
        <img src="data/images/realmdreamer.gif" alt="Teaser GIF showing 4 scenes - a bear, boat, bust in a museum, and the resolute desk">
    </picture>
</p>

This is the official implementation of the paper [RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion (3DV '25)](https://arxiv.org/abs/2404.07199)

> We introduce RealmDreamer, a technique for generating forward-facing 3D scenes from text descriptions. Our method optimizes a 3D Gaussian Splatting representation to match complex text prompts using pretrained diffusion models. As measured by a comprehensive user study, our method outperforms prior approaches, preferred by 88-95\%. We encourage viewing the supplemental website and video.


# Setup

1. Clone the repository and submodules:

```
git clone --recursive https://github.com/jaidevshriram/realmdreamer/
cd realmdreamer
git submodule update --init --recursive
```

2. Install the requirements:

```
conda create --name realmdreamer python=3.9
conda activate realmdreamer
bash ./setup.sh
```

3. Download the data from the paper:

```
cd outputs/
gdown --fuzzy https://drive.google.com/file/d/1eHo9ZxOgUdVqwSUe0vZR8byE7jSNYEhF/view?usp=sharing
unzip realmdreamer_outputs.zip
cd ..
```

Note: This installation was tested on CUDA 11.8. You may need to modify the installation script if you have a different version of CUDA.

We also provide a docker image for the project. You can pull the image using and then follow the instructions above.

# How to Run

If running one of the scenes from the paper, it is recommended to jump to stage 2 since the data provided includes results from stage 1.

<details>
<summary><h1>Stage 1 (Optional): Initialization</h1></summary>

Create a new config file in the configs folder. You can use `configs/resolute.yaml` as a template. Be sure to update the prompt and the auxillary prompt. If using a custom image, update the `base_img_path` field.

## Point cloud generation

Generate an initial point cloud based on the 

```
python main.py --config_path configs/resolute.yaml
```

## Pose Selection

There are two options here - either use a script to generate new poses or manually choose them. Currently, manually chosen poses can cover a wider baseline. For all scenes showcased in the paper, we already provide the initial data (including point clouds).

**Manual (Recommended)**: 

1. Open the viewer with the point cloud:

```
ns-train dummy-pcd --data outputs/resolute/init_transforms.json --pipeline.model.pcd_path outputs/resolute/pointcloud.ply --viewer.websocket-port 8008
```

2. Enter the render mode and move the camera around to select the poses. For every pose, be sure to "save" the pose.

3. When you have selected all poses, click on "render" to save the poses to a file. Name the file `extra_poses.json`.

Combine the original poses with the extra poses using:

```
python scripts/render_gsplat_from_ply.py --ply outputs/resolute/pointcloud.ply --init outputs/resolute/init_transforms.json --traj outputs/resolute/camera_paths/extra_poses.json
```

where `resolute` is the name of the scene.

</details>

# Stage 2 - Inpainting

Train the model using `run.sh`.

```
./run.sh <scene_name>
```

Example: `./run.sh resolute`

This will train the model and save it to `outputs/<scene_name>`. This will take a few hours and will require a GPU. By default, it will also log to weights and biases. You can modify training parameters in the `run.sh` script.

# Stage 3 - Finetuning

This stage is optional and mainly used to improve the overall quality of the scene.

1. Train a personalized text-to-image model with dreambooth using `./personalize.sh <path/to/scene>`. This will produce embeddings to improve the quality of the scene during finetuning.

2. Run the finetuning stage as 

```
./finetune.sh <scene_name>
```

Example: `./finetune.sh resolute`

This will save the model to `outputs/<scene_name>/finetuned`.

# Render Results

To render the results, you can load the checkpoint in the NeRFStudio viewer and choose a custom rendering trajectory. We already provide rendering trajectories for all scenes in the `outputs/<scene_name>/camera_paths/render.json` file.

```
./render.sh <path/to/saved/model/config> <path/to/camera_path_file> <path/to/output_render_file>
```

This will save the rendered video in the target folder.

# To Do

[] Automatic camera pose selection

# Citation

If you find our work interesting, please consider citing us!

```
    @inproceedings{shriram2024realmdreamer,
        title={RealmDreamer: Text-Driven 3D Scene Generation with 
                Inpainting and Depth Diffusion},
        author={Jaidev Shriram and Alex Trevithick and Lingjie Liu and Ravi Ramamoorthi},
        journal={International Conference on 3D Vision (3DV)},
        year={2025}
    }
```            

# Acknowledgements

 We thank Jiatao Gu and Kai-En Lin for early discussions, Aleksander Holynski and Ben Poole for later discussions. We thank Michelle Chiu for video design help. This work was supported in part by an NSF graduate Fellowship, ONR grant N00014-23-1-2526, NSF CHASE-CI Grants 2100237 and 2120019, gifts from Adobe, Google, Qualcomm, Meta, the Ronald L. Graham Chair, and the UC San Diego Center for Visual Computing. 

We also thank the authors of the following repositories for their code:

- [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) - used for training the model and rendering the results.
- [Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) - We build on the original 3DGS implementation.
- [Threestudio](https://github.com/threestudio-project/threestudio) - We adapt our distillation technique from similar implementations in Threestudio.
- [Pytorch3D](https://github.com/facebookresearch/pytorch3d) - We use Pytorch3D for the point cloud generation stage.
- [Diffusers](https://github.com/huggingface/diffusers) - Most models used are from the Diffusers library.
