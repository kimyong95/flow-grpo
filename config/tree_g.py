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

    # dataset + prompting
    config.dataset = os.path.join(os.getcwd(), "dataset/prompt_align_1")

    # total objective evaluations: 40*32*5*1=6400
    config.sample.total_num_samples = 32
    config.sample.batch_size = 1 # active set size A in the paper
    config.sample.expansion_size = 5 # branch out sample size K in the paper

    # rewards
    config.reward_fn = {"gemini": 1.0}

    return config


def get_config(name):
    return globals()[name]()
