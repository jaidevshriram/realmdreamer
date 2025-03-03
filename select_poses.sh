#!/bin/bash

# Check if the required arguments are provided
if [ $# -ne 1 ]; then
  echo "Usage: bash select_poses.sh [scene_name]"
  exit 1
fi

scene_name=$1

ns-train dummy-pcd --data outputs/$scene_name/init_transforms.json --pipeline.model.pcd_path outputs/$scene_name/pointcloud.ply --viewer.websocket-port 7007