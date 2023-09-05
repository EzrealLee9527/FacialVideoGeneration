import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import imageio

def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)
    inference_config = OmegaConf.load(args.inference_config)

    config  = OmegaConf.load(args.config)
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
        
            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.unet3d_pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
            if "state_dict" in motion_module_state_dict:
                motion_module_state_dict = motion_module_state_dict["state_dict"]
                motion_module_state_dict = {x.replace('module.',''):y for x,y in motion_module_state_dict.items()}
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            print(f'unet missing {len(missing)}, unexpected {len(unexpected)}')
            assert len(unexpected) == 0, ('unexpected:', unexpected)

            # adapter module
            adapter_module = model_config.adapter_module
            adapter_module_state_dict = torch.load(adapter_module, map_location="cpu")
            if "state_dict" in adapter_module_state_dict:
                adapter_module_state_dict = adapter_module_state_dict["state_dict"]
                adapter_module_state_dict = {x.replace('module.',''):y for x,y in adapter_module_state_dict.items()}
            from animatediff.models.adapter_module import Adapter
            model_adapter = Adapter(cin=64, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to("cuda")
            missing, unexpected = model_adapter.load_state_dict(adapter_module_state_dict)
            print(f'model_adapter missing {len(missing)}, unexpected {len(unexpected)}')
            # get landmarks
            # 待会儿写到config里
            from scripts.landmarks import get_landmarks
            import numpy as np
            video_path = '/dataset00/Videos/smile/gifs/smile_375.gif'
            # video_path = 'zy_datas/zy_wink_new.gif'
            frames_ldmks, frames = get_landmarks(video_path)
            frames_ldmks = torch.tensor(np.array(frames_ldmks)).permute(0,3,1,2).float().to("cuda")

            # tmp
            # imageio.mimsave('datas/zyzy_crop.gif', frames, fps=30)

            #
            img_path = '/work00/AnimateDiff-adapter/datas/30-1girl-is-smiling,-upper-body,-beautiful-face,-straight-hair,-long.gif'
            frames = imageio.mimread(img_path)
            image = frames[0] / 255
            image =  torch.Tensor((2.0 * np.array(image) - 1.0)).to("cuda")[:,:,:3]
            images = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).float()
            print('images', images.shape, images.max(), images.min())
            print('frames_ldmks', frames_ldmks.shape)
            encoding_dist = vae.encode(images).latent_dist
            encoding = encoding_dist.sample()
            image_latents = encoding * 0.18215
            base_noise_weight = np.sqrt(0.5) 
            res_noise_weight = np.sqrt(0.5) 
            base_noise = torch.randn_like(image_latents).repeat(args.L,1,1,1) * base_noise_weight
            image_latents = image_latents.repeat(args.L,1,1,1)
            print('image_latents', image_latents.shape, image_latents.max(), image_latents.min())
            noise = torch.randn_like(image_latents)
            res_noise = torch.randn_like(image_latents) * res_noise_weight
            noise = base_noise + res_noise
            batch_size = image_latents.shape[0]

            # save reconstruction results
            # latents_r = 1 / 0.18215 * image_latents
            # image_latents_reconstruction = vae.decode(latents_r[::2,:,:,:]).sample
            # print('image_latents_reconstruction', image_latents_reconstruction.shape)
            # image_latents_reconstruction = ((image_latents_reconstruction / 2 + 0.5).clamp(0, 1) * 255).permute(0,2,3,1).cpu().detach().numpy().astype('uint8')[:,:,::-1]
            # with imageio.get_writer(f"image_latents_reconstruction.mp4", fps=30) as video_writer:
            #     for frame in image_latents_reconstruction:
            #         video_writer.append_data(frame)
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            # 从模板加噪再去噪
            # get the original timestep using init_timestep
            num_inference_steps = model_config.steps
            pipeline.scheduler.set_timesteps(num_inference_steps, device='cuda')
            offset = pipeline.scheduler.config.get("steps_offset", 0)
            strength = args.strength
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = pipeline.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size, device='cuda')
            print('timesteps', timesteps)
            # add noise to latents using the timesteps
            image_latents = pipeline.scheduler.add_noise(image_latents, noise, torch.tensor(timesteps, dtype=torch.long))
            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = pipeline.scheduler.timesteps[t_start:].to('cuda')
            print('timesteps', timesteps)
            print("_____________",  image_latents.shape)
            image_latents = image_latents.unsqueeze(2)
            image_latents = rearrange(image_latents, "t c b h w -> b c t h w")
            print("_____________",  image_latents.shape)

            # eval using image format
            # f,c,h,w
            adapter_features = model_adapter(frames_ldmks)
            # transfer adapter_features to video format (b c f h w)
            adapter_features = [x.permute(1,0,2,3).unsqueeze(0).repeat(2,1,1,1,1) for x in adapter_features]
            print('adapter_features', len(adapter_features), adapter_features[0].shape)

            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                    
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            pipeline.to("cuda")
            ### <<< create validation pipeline <<< ###

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                    timesteps           = timesteps,
                    latents             = image_latents,
                    adapter_features    = adapter_features,
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                print(f"save to {savedir}/sample/{prompt}.gif")
                
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5",)
    parser.add_argument("--unet3d_pretrained_model_path", type=str, default='/huggingface00/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0',)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.8)

    args = parser.parse_args()
    main(args)
