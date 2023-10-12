import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from animatediff.models_2d.unet_2d_condition import UNet2DConditionModel
from animatediff.pipelines.pipeline_stable_diffusion_adapter import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print
import importlib
import numpy as np
import io
from scripts.landmarks import get_landmarks
from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
import cv2

def init_dist(launcher="slurm", backend='nccl', port=29501, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        print('launcher:pytorch')
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        print(port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank



def main(
    image_finetune: bool,
    adapter_model: Dict,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    s3_output_dir: str,
    pretrained_model_path: str,
    unet3d_pretrained_model_path: str,

    data_class: str,
    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,

    pretrained_adapter = None,

):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    s3_output_dir = os.path.join(s3_output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints/checkpoint-steps", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(unet3d_pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(unet3d_pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(unet3d_pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            unet3d_pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(unet3d_pretrained_model_path, subfolder="unet")
              
    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        with smart_open(unet_checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            unet_checkpoint_path = torch.load(buffer, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)
    # for name, param in unet.named_parameters():
    #     for trainable_module_name in trainable_modules:
    #         if trainable_module_name in name:
    #             param.requires_grad = True
    #             break

    # trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))

    # landmarks encoder
    from animatediff.models.adapter_module import Adapter
    model_adapter = Adapter(cin=adapter_model.get('cin', 64), channels=[320, 640, 1280, 1280][:4], 
                            nums_rb=2, ksize=1, sk=True, use_conv=False,
                            ckpt_path=pretrained_adapter).to(local_rank)
    model_adapter = DDP(model_adapter, device_ids=[local_rank], output_device=local_rank)
    trainable_params = list(model_adapter.parameters())
     
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    # train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    # train_dataset = WebVid(**train_data, is_image=image_finetune)
    
    dataset_cls = getattr(importlib.import_module('animatediff.data.datasets', package=None), data_class)
    train_dataset = dataset_cls(**train_data, is_image=image_finetune)
    print('train_dataset size is:', len(train_dataset))
    video_length = train_data['video_length']

    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # import pdb;pdb.set_trace()

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        # TODO: add adapter
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            unet3d_pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    # unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            transposed_list = list(zip(*batch['texts']))
            texts = [list(item) for item in transposed_list]
            
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values = batch['pixel_values'].cpu()
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text[0].replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text[0].replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")

            if cfg_random_null_text:
                texts = [name if random.random() > cfg_random_null_text_ratio else [''] * video_length for name in texts]
   
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            texts = [item for sublist in texts for item in sublist] 
            # print('debug:', len(texts), batch['landmarks'].shape, batch['pixel_values'].shape)
            if len(texts) == 0:
                texts = [''] * (train_batch_size * video_length)
            # print(texts)
            with torch.no_grad():
                prompt_ids = tokenizer(
                    texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            
            # adapter
            unet.zero_grad()
            adapter_input = batch['landmarks']
            # if image_finetune:
            #     mask_landmarks = np.random.randint(0,2)
            #     mask_face_parsings = np.random.randint(0,2)
            #     adapter_input = torch.concat((batch['landmarks']*mask_landmarks,batch['face_parsings']*mask_face_parsings), dim=1)
            # else:
            #     adapter_input = torch.concat((batch['landmarks']*mask_landmarks,batch['face_parsings']*mask_face_parsings), dim=2)
            adapter_features = model_adapter(adapter_input.to(local_rank))
                
            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, adapter_features=adapter_features).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and global_step % checkpointing_steps == 0:
                s3_save_dir = os.path.join(s3_output_dir, f"checkpoints")
                local_save_dir = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": model_adapter.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    local_save_path = os.path.join(local_save_dir, f"checkpoint-epoch-{epoch+1}.ckpt")
                    s3_save_path = os.path.join(s3_save_dir, f"checkpoint-epoch-{epoch+1}.ckpt")   
                else:
                    local_save_path = os.path.join(local_save_dir, f"checkpoint-epoch0-steps{global_step}.ckpt")
                    s3_save_path = os.path.join(s3_save_dir, f"checkpoint-epoch0-steps{global_step}.ckpt")
                torch.save(state_dict, local_save_path)
                # smart_sync(local_save_path, s3_save_path)
                os.system(f'aws --endpoint-url=https://tos-s3-cn-shanghai.ivolces.com s3 cp {local_save_path} {s3_save_path}')
                smart_remove(local_save_path)
                logging.info(f"Saved state to {s3_save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.resolution[0] if not isinstance(train_data.resolution, int) else train_data.resolution
                width  = train_data.resolution[1] if not isinstance(train_data.resolution, int) else train_data.resolution

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts
                # ada_rand_idx = np.random.randint(0,len(adapter_features)-1)

                
                
                # video_path = '/dataset00/Videos/smile/gifs/smile_23.gif'
                # frames_ldmks, frames = get_landmarks(video_path, video_length=8)
                # frame_ldmks = torch.tensor(np.array(frames_ldmks[-1])).permute(
                #     2, 0, 1).float().to("cuda").unsqueeze(0)
                # # f,c,h,w
                # test_adapter_features = model_adapter(frame_ldmks)
                # print('test_adapter_features', test_adapter_features.shape)
                # print('adapter_features[:1]', adapter_features[:1].shape)


                #####################################
                ldmks_video = '/work00/AnimateDiff-adapter-mmv2/templates/0927/107_The_person_turns_sad_into_sad_ldmks_frames16.mp4'
                cap = cv2.VideoCapture(ldmks_video)
                frames_ldmks = []  
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frames_ldmks.append(frame)
                    else:
                        break
                frames_ldmks = np.array(frames_ldmks) / 255
                save_frames_ldmks = torch.tensor(frames_ldmks).permute(3,0,1,2).unsqueeze(0).float()
                frames_ldmks = torch.tensor(frames_ldmks).permute(0,3,1,2).float().to("cuda")[:1,:1,:,:]
                # f,c,h,w
                adapter_features = model_adapter(frames_ldmks)
                test_adapter_features = [x.repeat(2,1,1,1) for x in adapter_features]
                print('adapter_features', len(adapter_features), adapter_features[0].shape)

                
                # test_adapter_features = [x.repeat(2,1,1,1)  for x in adapter_features]

                # one_adapter_features = [x[ada_rand_idx:ada_rand_idx+1].repeat(2,1,1,1) for x in adapter_features]
                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.video_length,
                            height       = height,
                            width        = width,
                            **validation_data,
                        ).videos
                        # print('sample', sample.shape)
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                        
                    else:     
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                            adapter_features    = test_adapter_features,
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("--local_rank", type=int, help="Local rank of the current process on the node")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
