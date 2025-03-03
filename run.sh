#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide the scene name as an argument"
  exit 1
fi

scene_name=$1

if [ -f "configs/${scene_name}.yaml" ]; then
  prompt=$(cat "configs/${scene_name}.yaml" | grep "prompt" | cut -d ":" -f 2 | head -1)
else
  echo "No config file found for scene '${scene_name}' named ${scene_name}.yaml"
  exit 1
fi

scene_folder_path="paper_outputs/${scene_name}"

command="ns-train realmdreamer --data "${scene_folder_path%/}" \
--project_name "RealmDreamer" \
--experiment_name '${scene_name}' \
--pipeline.prompt "${prompt}" \
--vis viewer+wandb \
--machine.num-devices 1 \
--save-only-latest-checkpoint True \
--steps_per_save 1000 \
--pipeline.datamanager.camera-optimizer.mode off \
--max-num-iterations 15000 \
--logging.steps_per_log 50 \
--viewer.websocket-port 7009 \
--viewer.quit-on-train-completion True \
--pipeline.datamanager.train-num-images-to-sample-from 1 \
--pipeline.datamanager.num_dilations 1 \
--gradient-accumulation-steps 1 \
--pipeline.datamanager.camera-res-scale-factor 1 \
--pipeline.datamanager.debug False \
--pipeline.densification-interval 250 \
--pipeline.model.guidance 'sds_inpainting' \
--pipeline.model.loss_type 'multi_nfsd' \
--pipeline.model.invert_ddim True \
--pipeline.model.use_sigmoid True \
--pipeline.model.average_colors False \
--pipeline.model.lambda_rgb 10000.0 \
--pipeline.model.lambda_depth 0.0 \
--pipeline.model.lambda_sds 0.1 \
--pipeline.model.lambda_opaque 0.0 \
--pipeline.model.lambda_one_step 0.01 \
--pipeline.model.lambda_one_step_perceptual 100 \
--pipeline.model.lambda_input_constraint_l2 1000 \
--pipeline.model.depth_guidance True \
--pipeline.model.depth_guidance_multi_step False \
--pipeline.model.depth_loss 'pearson' \
--pipeline.model.lambda_depth_sds 1000 \
--pipeline.input_view_constraint False \
--pipeline.input_view_depth_constraint False \
--pipeline.model.max_step_percent 0.98 \
--pipeline.model.min_step_percent 0.2 \
--pipeline.model.anneal False \
--pipeline.model.prolific_anneal False \
--pipeline.model.ignore_mask False \
--pipeline.model.img_guidance_scale 1.8 \
--pipeline.model.guidance_scale 7.5 \
--optimizers.xyz.optimizer.lr 0.01 \
--optimizers.f-dc.optimizer.lr 0.001 \
--optimizers.opacity.optimizer.lr 0.01 \
--optimizers.scaling.optimizer.lr 0.005 \
--optimizers.rotation.optimizer.lr 0.01 \
--pipeline.datamanager.split_mask_by_area_threshold False \
--pipeline.model.pcd_path "${scene_folder_path}/pointcloud.ply" "

echo $command
eval $command
