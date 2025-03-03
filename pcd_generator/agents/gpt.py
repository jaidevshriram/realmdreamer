import base64
import os
import pdb
import sys

import numpy as np
from base import Agent
from litellm import completion
from rich import print

sys.path.append("/mnt/data/temp/Portal")
from utils.pose import compare_poses


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GPTAgent(Agent):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.poses_list = np.load(cfg.poses_path)
        self.image_path = cfg.image_path
        print("Using GPT agent with preloaded poses")

        self.encoded_image = encode_image(self.image_path)

    def load_poses(self, poses_path):
        self.poses_list = np.load(poses_path)
        print("Loaded poses from", poses_path)

    def get_next_element(self):
        """Returns the ith pose from the list of poses."""

        if self._i < len(self.poses_list):
            return self.poses_list[self._i]
        else:
            return None

    def get_prompt(self, cur_pose, base_pose, prompt):

        base64_image = self.encoded_image

        print("Base pose:", base_pose)
        print("Current pose:", cur_pose)

        # Position and orientation of new pose relative to the current pose
        rel_pose = compare_poses(base_pose, cur_pose)
        rel_position = rel_pose["position"]
        rel_orientation = rel_pose["gaze"]

        print("Position:", rel_position)
        print("Orientation:", rel_orientation)

        custom_prompt = f"You are given an image of a {prompt} as viewed by an imaginary robot at some arbitrary position. The robot is currently looking forward. Now, imagine that it"

        # Add the relative position and orientation to the prompt
        if rel_position == "left" or rel_position == "right":
            custom_prompt += f" has moved to the {rel_position} and"
        elif rel_position == "up" or rel_position == "down":
            custom_prompt += f" has moved {rel_position} and"
        elif rel_position == "back" or rel_position == "forward":
            custom_prompt += f" has moved {rel_position} and"

        if rel_orientation == "left" or rel_orientation == "right":
            custom_prompt += f" is now looking to the {rel_orientation}."
        elif rel_orientation == "up" or rel_orientation == "down":
            custom_prompt += f" is now looking {rel_orientation}."

        custom_prompt += " What does the agent see now? Ensure that your answer is minimal and precise. Do not get too creative - the goal is to realistically capture the contents of the scene. For example, if you are looking at a sofa against an empty wall, it is better to say that there is a wall to the left/right of the sofa rather than saying that there is an entirely new room to the left/right of the sofa. The goal is to describe the adjacent environment in a manner that keeps the focus of the scene on the provided image."

        print(custom_prompt)

        # openai call
        # response = completion(
        #     model = "gpt-4-vision-preview",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                             {
        #                                 "type": "text",
        #                                 "text": "What’s in this image?"
        #                             },
        #                             {
        #                                 "type": "image_url",
        #                                 "image_url": {
        #                                     "url": f"data:image/jpeg;base64,{base64_image}"
        #                                 }
        #                             }
        #                         ]
        #         }
        #     ],
        # )

        return f"Given the image of a {cur_pose}, what is the next pose?"


if __name__ == "__main__":

    from collections import namedtuple

    cfg = {
        "poses_path": "/mnt/data/temp/Portal/data/poses/look_left_right_back_left_right.npy",
        "image_path": "/mnt/data/temp/Portal/data/images/astronaut2.jpg",
    }

    cfg = namedtuple("Config", cfg.keys())(*cfg.values())

    agent = GPTAgent(cfg)

    pdb.set_trace()

    agent.get_prompt(agent.poses_list[1], agent.poses_list[0], "astronaut in a cave")
