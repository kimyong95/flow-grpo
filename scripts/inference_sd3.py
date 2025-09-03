from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
import accelerate
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_opt import pipeline_with_logprob, latents_to_images
from flow_grpo.diffusers_patch.sd3_sde_with_logprob_opt import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
import einops
from functools import partial
import tqdm
import torch.nn as nn
import math
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from value_model import ValueModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Union, List, Callable, Dict, Any

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return (idx, self.prompts[idx], {})
    
    @staticmethod
    def collate_fn(batch):
        # batch: list of (idx, prompt_str, metadata_dict)
        idxs, prompts, metas = zip(*batch)
        idxs = list(idxs)
        prompts = list(prompts)
        metas = list(metas)
        return idxs, prompts, metas
    
class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return (idx, self.prompts[idx], self.metadatas[idx])

    @staticmethod
    def collate_fn(batch):
        # batch: list of (idx, prompt_str, metadata_dict)
        idxs, prompts, metas = zip(*batch)
        idxs = list(idxs)
        prompts = list(prompts)
        metas = list(metas)
        return idxs, prompts, metas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


def sde_step_with_expansion(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    noise: Optional[torch.FloatTensor] = None,
    expansion_size_t: List[int] = None,
    reward_fn = None,
    sample_one_step_fn = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    pred_sample = (sample - sigma * model_output)
    
    return prev_sample, None, prev_sample_mean, std_dev_t, pred_sample

def sample_one_step(
    self,
    latents,
    t,
    prompt_embeds,
    pooled_prompt_embeds,
):
    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
    timestep = t.expand(latent_model_input.shape[0])
    noise_pred = self.transformer(
        hidden_states=latent_model_input.to(self.transformer.dtype),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=self.joint_attention_kwargs,
        return_dict=False,
    )[0]
    noise_pred = noise_pred.to(prompt_embeds.dtype)
    if self.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    _, _, _, _, pred_sample = sde_step_with_logprob(
        self.scheduler, 
        noise_pred.float(), 
        t.unsqueeze(0), 
        latents.float(),
        noise_level=0.0,
        noise=None,
    )

    return pred_sample

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
            name=f"inference-{config.run_name}",
            config=config.to_dict(),
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=torch.bfloat16,
    )
    pipeline.vae.enable_slicing()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    if config.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)
    pipeline.transformer.eval()

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move to inference_dtype
    pipeline.to(accelerator.device, dtype=inference_dtype)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.init_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.init_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.init_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.init_batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    train_dataloader = accelerator.prepare(train_dataloader)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    
    dimension = pipeline.transformer.config.in_channels * int(config.resolution/pipeline.vae_scale_factor) * int(config.resolution/pipeline.vae_scale_factor)
    mu = torch.zeros((config.sample.num_steps, dimension), device=accelerator.device)
    sigma = torch.ones((config.sample.num_steps, dimension), device=accelerator.device)

    flatten = lambda x: einops.rearrange(x, "... c h w -> ... (c h w)", c=pipeline.transformer.config.in_channels, h=int(config.resolution/pipeline.vae_scale_factor), w=int(config.resolution/pipeline.vae_scale_factor))
    unflatten = lambda x: einops.rearrange(x, "... (c h w) -> ... c h w", c=pipeline.transformer.config.in_channels, h=int(config.resolution/pipeline.vae_scale_factor), w=int(config.resolution/pipeline.vae_scale_factor))

    value_model = ValueModel(dimension=dimension)
    value_model.to(accelerator.device)

    # b_t in the paper
    batch_size_t = [
        int( config.sample.init_batch_size * (config.sample.final_batch_size/config.sample.init_batch_size)**(t/config.sample.num_steps) )
        for t in range(config.sample.num_steps)
    ]

    # w_t in the paper
    expansion_size_t = [
        int(config.sample.evaluation_budget // b_t)
        for b_t in batch_size_t
    ]

    objective_evaluations = [
        b*e
        for b,e in zip(batch_size_t, expansion_size_t)
    ]
    objective_evaluations = torch.tensor(objective_evaluations).cumsum(dim=0)

    train_sampler.set_epoch(0)
    prompts_idx, prompts, prompt_metadata = next(train_iter)

    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        [prompts[0]]*config.sample.init_batch_size, 
        text_encoders, 
        tokenizers, 
        max_sequence_length=128, 
        device=accelerator.device
    )

    def callback_fn(self, index, timestep, kwargs):

        if index == self.scheduler.num_inference_steps - 1:
            return {}

        timesteps = kwargs["timesteps"]
        prev_latents_mean = kwargs["prev_latents_mean"]
        std_dev_t = kwargs["std_dev_t"]
        prompt_embeds = kwargs["prompt_embeds"]
        prompt_embeds_base = torch.stack([prompt_embeds[0], prompt_embeds[-1]])
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]
        pooled_prompt_embeds_base = torch.stack([pooled_prompt_embeds[0], pooled_prompt_embeds[-1]])
        
        prev_step_index = index + 1
        sigma = pipeline.scheduler.sigmas[index].view(-1, *([1] * (len(prev_latents_mean.shape) - 1)))
        sigma_prev = pipeline.scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(prev_latents_mean.shape) - 1)))
        sigma_max = pipeline.scheduler.sigmas[1].item()
        dt = sigma_prev - sigma
        prev_timestep = timesteps[prev_step_index]

        batch_size = batch_size_t[index]
        expansion_size = expansion_size_t[index]

        prompt_embeds_expand = einops.repeat(
            prompt_embeds_base,
            '(two) ... -> (two m) ...', two=2, m = expansion_size,
        )

        pooled_prompt_embeds_expand = einops.repeat(
            pooled_prompt_embeds_base,
            '(two) ... -> (two m) ...', two=2, m = expansion_size,
        )
        
        prev_sample = []
        prev_sample_rewards = []
        images = []
        for i, prev_latents_mean_i in enumerate(prev_latents_mean):

            noise_i = randn_tensor(
                (expansion_size,) + tuple(prev_latents_mean_i.shape),
                dtype=prev_latents_mean_i.dtype,
                device=prev_latents_mean_i.device,
            )
            prev_sample_i = prev_latents_mean_i + std_dev_t * torch.sqrt(-1*dt) * noise_i
            pred_sample_i = sample_one_step(
                self,
                latents=prev_sample_i,
                t=prev_timestep,
                prompt_embeds=prompt_embeds_expand,
                pooled_prompt_embeds=pooled_prompt_embeds_expand,
            )
            images_i = latents_to_images(self, pred_sample_i, output_type="pt")
            _prompts = [prompts[0]] * expansion_size
            _prompts_metadata = [prompt_metadata[0]] * expansion_size
            rewards_i, rewards_meta_i = reward_fn(images_i, _prompts, _prompts_metadata)
            best_i = torch.argmax(torch.tensor(rewards_i["avg"])).item()
            prev_sample.append(prev_sample_i[best_i])
            prev_sample_rewards.append({key: value[best_i] for key, value in rewards_i.items()})
            images.append(images_i[best_i])

        # aggregate
        prev_sample = torch.stack(prev_sample)
        prev_sample_rewards = {k: torch.tensor([dic[k] for dic in prev_sample_rewards]) for k in prev_sample_rewards[0]}
        
        # filter
        next_batch_size = batch_size_t[prev_step_index]
        next_indices = prev_sample_rewards["avg"].topk(next_batch_size).indices
        prev_sample_rewards = {k: v[next_indices] for k, v in prev_sample_rewards.items()}
        prev_sample = prev_sample[next_indices]
        prev_sample = prev_sample.to(self.transformer.dtype)
        images = [images[idx] for idx in next_indices]
        prompt_embeds = einops.repeat(prompt_embeds_base, '(two) ... -> (two b) ...', two=2, b = next_batch_size)
        pooled_prompt_embeds = einops.repeat(pooled_prompt_embeds_base, '(two) ... -> (two b) ...', two=2, b = next_batch_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(len(images)):
                image = images[i]
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))

            wandb.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompts[0]} | avg: {avg_reward:.2f}",
                        )
                        for idx, avg_reward in enumerate(prev_sample_rewards["avg"])
                    ],
                    "objective_evaluations": objective_evaluations[index],
                    **{f"reward_{key}": value.mean() for key, value in prev_sample_rewards.items()},
                },
                step=index,
            )

        return {
            "latents": prev_sample,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    with autocast():
        with torch.inference_mode():
            
            callback_fn_inputs = ["timesteps", "prev_latents_mean", "std_dev_t", "prompt_embeds", "pooled_prompt_embeds"]
            pipeline._callback_tensor_inputs.extend(callback_fn_inputs)
            images, latents, log_probs, pred_samples = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                output_type="pt",
                height=config.resolution,
                width=config.resolution, 
                noise_level=config.sample.noise_level,
                callback_on_step_end=callback_fn,
                callback_on_step_end_tensor_inputs=callback_fn_inputs,
            )
            images = images.to(accelerator.device, dtype=torch.float32)

    rewards, rewards_meta = reward_fn(
        images,
        [prompts[0]]*config.sample.final_batch_size,
        [prompt_metadata[0]]*config.sample.final_batch_size,
    )

    pil_images = [
        Image.fromarray(
            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        ).resize((config.resolution, config.resolution))
        for image in images
    ]    

    with tempfile.TemporaryDirectory() as tmpdir:
        wandb.log(
            {
                "images": [
                    wandb.Image(
                        image,
                        caption=f"{prompts[0]} | avg: {avg_reward:.2f}",
                    )
                    for image, avg_reward in zip(pil_images, rewards["avg"])
                ],
                "objective_evaluations": objective_evaluations[-1],
                **{f"reward_{key}": torch.tensor(value).mean() for key, value in rewards.items()},
            },
            step=global_step,
        )

if __name__ == "__main__":
    app.run(main)