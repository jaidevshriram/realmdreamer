#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_file> <camera_path_filename> <video_filename>"
    exit 1
fi

# Extract the scene name from the config file path
scene_name=$(basename "$(dirname "$(dirname "$(dirname "$1")")")")
video_name=$(basename "$3")

# Create the output path
output_path="renders/$scene_name/$video_name.mp4"

# Create the command
command="ns-render camera-path --load-config $1 --camera-path-filename $2 --output-format video --output-path $output_path --rendered-output-names rgb --colormap-options.normalize True --colormap-options.specified_normalize False"

# Print and execute the command
echo "Running command:"
echo "$command"
eval "$command"
