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

    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"jpeg_compressibility": 1}

    return config

def prompt_align():
    config = compressibility()

    config.run_name = "prompt-align"

    # dataset
    config.dataset = os.path.join(os.getcwd(), "dataset/prompt_align_1")

    # rewards
    config.reward_fn = {"gemini": 1.0}

    # algorithm-specific setting
    # total objective evaluation ≈ 180 * 40 = 7200
    # but actually due to rounding it is ≈ 6400
    config.sample.init_batch_size = 64     # b_0 in the paper
    config.sample.final_batch_size = 32    # b_1 in the paper
    config.sample.evaluation_budget = 180  # C in the paper

    return config

def get_config(name):
    return globals()[name]()
