#!/bin/bash

# Require the scene name as an argument
if [ -z "$1" ]; then
  echo "Please provide the scene name as an argument"
  exit 1
fi

scene_name=$1

if [ -f "configs/${scene_name}.yaml" ]; then
  prompt=$(cat "configs/${scene_name}.yaml" | grep "prompt" | cut -d ":" -f 2 | head -1)
else
  echo "No config file found for scene: ${scene_name}"
  exit 1
fi

# Find the most recent checkpoint in the folder
ckpt_name=$(ls -t "outputs/${scene_name}/realmdreamer" | head -1)

echo "Finetuning for scene: ${scene_name}"

scene_folder_path="outputs/${scene_name}"

ckpt_name="2025-01-26_053355"

command="ns-train realmdreamer-f --data "${scene_folder_path%/}" \
--project_name "RealmDreamer_Finetune" \
--experiment_name "${scene_name}" \
--pipeline.prompt "${prompt}" \
--vis wandb \
--load-dir "outputs/${scene_name}/realmdreamer/${ckpt_name}/nerfstudio_models" \
--machine.num-devices 1 \
--save-only-latest-checkpoint True \
--steps_per_save 1000 \
--pipeline.datamanager.camera-optimizer.mode off \
--max-num-iterations 3000 \
--logging.steps_per_log 50 \
--viewer.websocket-port 7009 \
--viewer.quit-on-train-completion True \
--pipeline.datamanager.train-num-images-to-sample-from 1 \
--pipeline.datamanager.num_dilations 1 \
--gradient-accumulation-steps 1 \
--pipeline.datamanager.camera-res-scale-factor 1 \
--pipeline.datamanager.debug False \
--pipeline.density_start_iter 0 \
--pipeline.densification-interval 250 \
--pipeline.model.guidance 'sds' \
--pipeline.model.loss_type 'multi_step' \
--pipeline.model.load_dreambooth True \
--pipeline.model.use_sigmoid True \
--pipeline.model.average_colors False \
--pipeline.model.invert_ddim True \
--pipeline.model.invert-after-step False \
--pipeline.model.invert_step_ratio 0.3 \
--pipeline.model.ddim_invert_method 'ddim_0' \
--pipeline.model.num_steps_sample 100 \
--pipeline.model.fixed_num_steps False \
--pipeline.model.lambda_rgb 0.0 \
--pipeline.model.lambda_depth 0.0 \
--pipeline.model.lambda_sds 0.01 \
--pipeline.model.lambda_opaque 10.0 \
--pipeline.model.lambda_one_step 0.01 \
--pipeline.model.lambda_one_step_l1 0.0 \
--pipeline.model.lambda_one_step_perceptual 100 \
--pipeline.model.lambda_one_step_ssim 100 \
--pipeline.model.sharpen_in_post False \
--pipeline.model.sharpen_in_post_factor 2.0 \
--pipeline.model.depth_guidance False \
--pipeline.model.depth_guidance_multi_step False \
--pipeline.model.load_depth_guidance True \
--pipeline.model.depth_loss 'pearson' \
--pipeline.model.lambda_depth_sds 100 \
--pipeline.input_view_constraint False \
--pipeline.input_view_depth_constraint False \
--pipeline.input_view_depth_constraint_type 'pearson' \
--pipeline.model.lambda_input_constraint_l2 1000 \
--pipeline.model.max_step_percent 0.3 \
--pipeline.model.min_step_percent 0.1 \
--pipeline.model.anneal False \
--pipeline.model.ignore_mask True \
--pipeline.model.img_guidance_scale 1.2 \
--pipeline.model.guidance_scale 7.5 \
--optimizers.f-dc.optimizer.lr 0.005 \
--optimizers.opacity.optimizer.lr 0.01 \
--optimizers.scaling.optimizer.lr 0.001 \
--optimizers.rotation.optimizer.lr 0.001 \
--pipeline.datamanager.split_mask_by_area_threshold False \
--pipeline.model.pcd_path "${scene_folder_path}/pointcloud.ply" "
echo $command
eval $command
