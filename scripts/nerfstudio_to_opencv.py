import argparse
import json
import os
import pdb

import numpy as np


def str_to_np(str):
    """
    Converts a string of the form '[1,2,3,4]' to a numpy array of the form [1,2,3,4]
    """
    str = str.replace("[", "")
    str = str.replace("]", "")
    str = str.split(",")
    str = [float(x) for x in str]
    return np.array(str)


def pose_to_opencv(pose):
    transform = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    return transform @ pose


def rotatey180(pose):

    transform = np.array(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    return pose @ transform


def rotatez180(pose):

    transform = np.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    return pose @ transform


def convert(json_file, data_parser_file, output_folder):
    """
    The nerfstudio camrea trajectory JSON looks something like -
    - keyframes: [
        {
            "matrix": "[1,0,0,0,0,2.220446049250313e-16,1,0,0,-1,2.220446049250313e-16,0,0.2796187400817871,1.7763568394002505e-15,-0.012583178468048795,1]",
            "fov": 90,
            "aspect": 1,
            "properties": "[[\"FOV\",90],[\"NAME\",\"Camera 0\"],[\"TIME\",0]]"
        }
    ]  # This is a list of keyframes
    - render_height
    - render_width
    - fps
    .....
    - camera_path: [
        {
            "camera_to_world": [
                1,
                0,
                0,
                0.2796187400817871,
                0,
                2.220446049250313e-16,
                -1,
                1.7763568394002505e-15,
                0,
                1,
                2.220446049250313e-16,
                -0.012583178468048795,
                0,
                0,
                0,
                1
            ],
            "fov": 90,
            "aspect": 1
        }
    ] # This is the actual list of camera poses used for rendering
    """

    file_name = os.path.basename(json_file).split(".")[0]

    input = json.load(open(json_file, "r"))

    # For the output json, we only need to select every skip-th camera path keyframe
    output_json = input.copy()
    output_json["camera_path"] = input["camera_path"]

    # Save the output json - ONLY for camera rendering - to use in a training pose, we need to fix scale and coordinate frame
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    # with open(os.path.join(output_folder, 'inpaint_path_nerfstudio.json'), 'w') as f:
    #     json.dump(output_json, f, indent=4)

    # Load dataparser to convert from NeRF scale to pointcloud's scale
    if data_parser_file is not None:
        data_parser = json.load(open(data_parser_file, "r"))

        transform = data_parser["transform"]  # 3 x 4 matrix
        transform = np.array(transform)
        # APpend the last row
        transform = np.vstack((transform, np.array([0, 0, 0, 1])))

        scale = data_parser["scale"]
    else:
        transform = np.eye(4)
        scale = 1

    output_keyframes = []

    # Convert each keyframeq
    for keyframe in input["camera_path"]:

        keyframe_matrix = keyframe["camera_to_world"]

        # Convert the matrix to a numpy array

        # if keyframe - uncomment
        # keyframe_matrix = str_to_np(keyframe_matrix).reshape(4, 4).T

        # if camera path - uncomment
        keyframe_matrix = np.array(keyframe_matrix).reshape(4, 4)

        # Scale the translation
        keyframe_matrix[:3, 3] /= scale

        # Multiply with transform
        keyframe_matrix = np.linalg.inv(transform) @ keyframe_matrix

        # idk why we do this honestly
        keyframe_matrix[:, 1:3] *= -1

        # # Convert to opencv format
        keyframe_matrix = pose_to_opencv(keyframe_matrix)

        # # Convert from c2w to w2c
        keyframe_matrix = np.linalg.inv(keyframe_matrix)

        # # Rotate the FRAME around Z by 180 degrees (again, idk why tbh)
        keyframe_matrix = rotatez180(keyframe_matrix)

        # Rotate teh FRAME around y by 180 degrees
        keyframe_matrix = rotatey180(keyframe_matrix)

        output_keyframes.append(keyframe_matrix)

    # Save the output keyframes
    np.save(os.path.join(output_folder, f"extra_poses.npy"), np.array(output_keyframes))

    print("Saved poses from camera path")

    return np.array(output_keyframes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert nerf studio camera path to pytorch3D format - exports a npy file\n"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="input file - the camera_path exported from the viewer",
        required=True,
    )
    parser.add_argument(
        "--dataparser",
        type=str,
        help="dataparser file - saved alongside the model weights",
        required=False,
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="output folder - where you want to save the poses",
        required=True,
    )

    args = parser.parse_args()

    output = convert(args.input, args.dataparser, args.output_folder)
