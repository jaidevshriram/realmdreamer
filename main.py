import argparse
import base64
import os
import subprocess
import time

from cprint import cprint
from litellm import completion

from configs import get_cfg_defaults
from pcd_generator.generator import Generator


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main(cfg):

    cprint.info("Starting Experiment: {}".format(cfg.project_name))
    cprint.info("Starting at: {}".format(time.ctime()))

    # Stage 1: Point Cloud Based Scene Generation

    generator = Generator(cfg)

    # Generate scene only if we aren't importing something new
    if not cfg.import_scene:
        final_pcd = generator.start_generation(cfg.base_img_path)

    generator.export_to_dataset()


if __name__ == "__main__":

    # Set OS environ key
    os.environ["OPENAI_API_KEY"] = "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"

    args = argparse.ArgumentParser()
    args.add_argument(
        "--config_path", help="Path to Config File", required=True, default=""
    )
    args, _ = args.parse_known_args()

    # Load config file
    cfg = get_cfg_defaults()

    if (
        os.path.exists(args.config_path)
        and os.path.splitext(args.config_path)[1] == ".yaml"
    ):
        cfg.merge_from_file(args.config_path)
    else:
        print("No valid config specified")
        exit(1)

    # Get the auxillary prompt from GPT-4
    base64_image = encode_image(cfg.base_img_path)
    input_prompt = cfg.prompt

    # prompt = f"""You are given an image of a {input_prompt} as viewed by an imaginary robot at some arbitrary position. The robot is currently looking forward. Now, imagine that it takes a step back and looks to the left and right, I want you to provide a prompt that describes what it sees.  You must follow multiple criteria to ensure that your answer is valid:

    # 1. If the image is that of an outdoor scene, then describe what I might see to the left and right of the scene. For example, if I am looking at a scene consisting of a meadow or a beach, then output a description of the meadow or the beach. However, the prompt should not describe things that are both near and far away from the robot, just the most important things that can describe the scene.

    # 2. If the image is that of an indoor scene, identify what the background elements of this indoor scene might be. For example, if the image is that of a living room, then identify the colour of the walls, the floor, and provide a description of the same. 

    # When you are providing your description for either case, I want you to follow the following rules:

    # 1. Do not be verbose or use full sentences, just use phrases
    # 2. There should not be any objects in the prompt you provide - do not add furniture, lamps, or any other objects to the prompt.
    # 3. There should not be any humans or people in the prompt you provide
    # 4. Keep your answer short and only output the phrases that describe the scene, not the entire sentence and defintely don't be verbose. If the original prompt describes a style such as 'photorealistic', 'watercolor', 'anime', etc, you should include those in the end of the prompt. 
    # 5. Your answer should not have words like 'left' or 'right' - it should be a general description, almost like an auxiliary description of the scene.

    # What is your answer? Output the answer only.
    # """

    # # openai call
    # response = completion(
    #     model="gpt-4-vision-preview",
    #     max_tokens=100,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
    #                 },
    #             ],
    #         }
    #     ],
    # )

    # cfg.aux_prompt = response.choices[0].message.content

    cprint.info(cfg)

    main(cfg)
