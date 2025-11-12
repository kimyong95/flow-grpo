


import glob
import re
import numpy as np
import yaml
import textwrap
import shutil
import os
import io
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import importlib
import wandb
from enum import Enum
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))

# Create a parser object
parser = argparse.ArgumentParser(description="A script to demonstrate taking arguments.")
parser.add_argument("--demo-id", type=int, default=0, help="Demo ID.")
args = parser.parse_args()

all_images = defaultdict(list)


cache_path = f"{dir_path}/wandb_cache.pkl"
with open(cache_path, 'rb') as f:
    wandb_cache = pickle.load(f)

def random_image():
    return Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8))

api = wandb.Api()
wandb_path = "kimyong95/flow_grpo"

missing = []

######################## PPTX ########################
from pptx import Presentation
from pptx.util import Inches
from pptx.util import Pt
from pptx.enum.shapes import PP_PLACEHOLDER

ppt = Presentation(f"{dir_path}/visualize/visualize-template.pptx")

# sorted x, y
def get_placeholders(slide):
    # Initialize lists for image and text placeholders
    image_placeholders = []
    text_placeholders = []

    # Iterate through shapes in the slide
    for shape in slide.shapes:
        if shape.is_placeholder:
            placeholder_type = shape.placeholder_format.type

            # Check placeholder type
            if placeholder_type == PP_PLACEHOLDER.PICTURE:  # PP_PLACEHOLDER.PICTURE (6) or PP_PLACEHOLDER.CONTENT (7)
                image_placeholders.append(shape)
            elif placeholder_type == PP_PLACEHOLDER.BODY:  # PP_PLACEHOLDER.TITLE (1), SUBTITLE (2), TEXT (3, 4)
                text_placeholders.append(shape)
    image_placeholders.sort(key=lambda x: (x.top, x.left))
    text_placeholders.sort(key=lambda x: (x.top, x.left))

    return image_placeholders, text_placeholders

def fill_contents(slide, images, texts):
    image_placeholders, text_placeholders = get_placeholders(slide)
    assert len(image_placeholders) == len(images)
    assert len(text_placeholders) == len(texts)
    for image_placeholder, image in zip(image_placeholders, images):
        if image is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_placeholder.insert_picture(buffer)
    for text_placeholder, text in zip(text_placeholders, texts):
        if type(text) == str:
            text_placeholder.text = text
        elif type(text) == list:
            text_frame = text_placeholder.text_frame
            for i, _text in enumerate(text):
                if i > 0:
                    text_frame.add_paragraph()
                p = text_frame.paragraphs[i]
                p.text = _text["text"]
                p.font.size = Pt(_text["fontsize"])

######################## PPTX ########################


#################### load images #####################
algos = ["dno", "grpo", "optimize"]
prompt_ids = [1,2,3,4,5,6]
run_name_format = "{algo}-prompt-align-{prompt_id}"

visualize_objective_evaluations = [0,1280,2560,3840,5120,6400]
algo_image_key = {
    # "d-search": "images",
    # "tree-g": "images",
    "grpo": "eval_noisy_images",
    "dno": "images",
    "optimize": "eval_images",
}

demo_id = args.demo_id
all_images = defaultdict(list)

for prompt_id in prompt_ids:
    for algo in algos:
        run_name = run_name_format.format(algo=algo, prompt_id=prompt_id)
        image_dir = f"{dir_path}/wandb-images/{run_name}"

        wandb_images = wandb_cache[run_name]["history"][algo_image_key[algo]]
        objective_evaluations = wandb_cache[run_name]["history"]["objective_evaluations"]
        assert objective_evaluations[objective_evaluations.notna()].is_monotonic_increasing
        
        obj_image = pd.DataFrame({
            "obj": objective_evaluations,
            "img": wandb_images.to_numpy(),
        })
        obj_image["obj"] = obj_image["obj"].ffill()
        obj_image = obj_image[obj_image["img"].notna()]
        obj_image = obj_image.set_index('obj').reindex(visualize_objective_evaluations, method='nearest')

        wandb_run = api.run(f"{wandb_path}/{wandb_cache[run_name]['run_id']}")

        # iterate over objective evaluations
        for wandb_image in obj_image["img"]:

            image_file = wandb_image["filenames"][demo_id]
            image_path = f"{image_dir}/{image_file}"
            
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                print(f"Downloading files [{run_name}] ...")
                wandb_run.file(name=image_file).download(image_dir)
                image = Image.open(image_path)

            all_images[run_name].append(image)
#################### load images #####################


################### create slides ####################

for prompt_id in prompt_ids:
    layout = ppt.slide_layouts[0]
    slide = ppt.slides.add_slide(layout)
    slide_images = []
    slide_texts = []
    for algo in algos:
        run_name = run_name_format.format(algo=algo, prompt_id=prompt_id)
        slide_images.extend(all_images[run_name])
        slide_texts.append(algo.upper())
    fill_contents(slide, slide_images, slide_texts)
ppt.save(f"{dir_path}/visualize/visualize-data={demo_id}.pptx")

################### create slides ####################