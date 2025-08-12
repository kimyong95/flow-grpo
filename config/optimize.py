import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def ocr():
    config = base.get_config()

    config.run_name = "ocr"

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.num_batches_per_epoch = 1
    config.sample.train_batch_size = 32

    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_1")

    config.prompt_fn = "general_ocr"

    config.reward_fn = {"ocr": 1.0,}

    return config


def compressibility():
    config = base.get_config()

    config.run_name = "compressibility"

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.num_batches_per_epoch = 1
    config.sample.train_batch_size = 32

    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.prompt_fn = "general_ocr"

    config.reward_fn = {"jpeg_compressibility": 1}

    return config

def get_config(name):
    return globals()[name]()
