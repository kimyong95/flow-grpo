import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()
    config.max_epochs = 500
    config.eval_freq = 10

    config.run_name = "compressibility"
    config.compile = True

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.resolution = 512

    config.sample.num_batches_per_epoch = 1
    config.sample.train_batch_size = 32
    config.sample.test_batch_size = 32

    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.prompt_fn = "general_ocr"

    config.reward_fn = {"jpeg_compressibility": 1}

    return config

def prompt_align():
    config = base.get_config()
    config.max_epochs = 500
    config.eval_freq = 10

    config.run_name = "prompt-align"
    config.compile = True

    # model + sampling
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.resolution = 512

    # batching (keep consistent with other optimize presets)
    config.sample.num_batches_per_epoch = 1
    config.sample.train_batch_size = 32
    config.sample.test_batch_size = 32

    # dataset + prompting
    config.dataset = os.path.join(os.getcwd(), "dataset/prompt_align_1")
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"gemma": 1.0}

    return config

def get_config(name):
    return globals()[name]()
