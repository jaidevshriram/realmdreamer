#!/bin/bash

# This script will train dreambooth on a given scene and then save the results to a folder

# Check if the number of arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <scene_path> [ex: $0 outputs/resolute/]"
    exit 1
fi

scene_path=$1
scene_name=$(basename $1)

if [ -f "configs/${scene_name}.yaml" ]; then
  prompt=$(cat "configs/${scene_name}.yaml" | grep "prompt" | cut -d ":" -f 2 | head -1)
else
  echo "No config file found for scene: ${scene_name}"
  exit 1
fi

class_prompt=$prompt
instance_prompt="${prompt}, in the style of <token>"

model_name="stabilityai/stable-diffusion-2-1-base"

rgb_dir="${scene_path}/input_rgb"
output_dir="${scene_path}/dreambooth"

python scripts/train_dreambooth.py \
  --mixed_precision fp16 \
  --pretrained_model_name_or_path=$model_name  \
  --instance_data_dir $rgb_dir \
  --instance_prompt "$instance_prompt" \
  --class_prompt "$class_prompt" \
  --output_dir=$output_dir \
  --resolution=512 \
  --train_batch_size=2 \
  --train_text_encoder \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=500 \
  --validation_steps=40 \
  --max_train_steps=200 \
  --use_8bit_adam
