import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.run_name = "compressibility"
    config.compile = True

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.guidance_scale = 4.5
    config.resolution = 512

    config.sample.mini_batch_size = 16     # Not in paper
    config.sample.init_batch_size = 64     # b_0 in the paper
    config.sample.final_batch_size = 32    # b_1 in the paper
    config.sample.evaluation_budget = 64   # C in the paper

    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.prompt_fn = "general_ocr"

    config.reward_fn = {"jpeg_compressibility": 1}

    return config

def prompt_align():
    config = compressibility()

    config.run_name = "prompt-align"

    # dataset + prompting
    config.dataset = os.path.join(os.getcwd(), "dataset/prompt_align_1")
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"gemma": 1.0}

    return config

def pickscore():
    config = compressibility()

    config.run_name = "pickscore"

    config.sample.num_image_per_prompt = 32

    # dataset + prompting
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.eval_freq = 0

    # rewards
    config.reward_fn = {"pickscore": 1.0}

    return config

def get_config(name):
    return globals()[name]()
