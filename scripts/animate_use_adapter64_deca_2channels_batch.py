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
from animatediff.data.datasets_deca import read_vis, read_vis_single
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
import numpy as np
import cv2

from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
import io


def get_adapter_input(
    video_id,
    lora_id,
    start_idx,
    frame_stride, 
    video_length,
    val_data_prefix,
    resolution_w,
    resolution_h,
    con1_wo_texture,
    con2_texture,
    con3_depth,
    cin, 
    deca_replace,
    savedir,
):
    # 先获取 adapter features
    # NOTE 其实这部分可以写在循环外
    org_img_list = []
    con_1_list = []
    con_2_list = []
    depth_list = []

    for frame_idx in range(start_idx, start_idx + frame_stride * video_length, frame_stride):
        if lora_id:
            val_vid_prefix=os.path.join(val_data_prefix, str(video_id),str(frame_idx),lora_id)
        else:
            val_vid_prefix=os.path.join(val_data_prefix, str(video_id),str(frame_idx))
        if deca_replace == False:
            org_img, con_1, con_2, depth = read_vis(os.path.join(val_data_prefix, str(video_id), str(frame_idx), "vis.jpg"), resolution_w)
        else:
            # org_img, _, _, _ = read_vis(os.path.join(val_data_prefix, str(video_id), "deca_v231023", str(frame_idx), "vis.jpg"), resolution_w)
            org_img = read_vis_single(os.path.join(val_data_prefix, str(video_id), str(frame_idx), "orig_inputs.jpg"), resolution_w,resolution_h)
            con_1 = read_vis_single(os.path.join(val_vid_prefix, "orig_shape_images.jpg"), resolution_w,resolution_h)
            con_2 = read_vis_single(os.path.join(val_vid_prefix, "orig_shape_detail_images.jpg"), resolution_w,resolution_h)
            depth = read_vis_single(os.path.join(val_data_prefix, str(video_id), str(frame_idx), "depth.jpg"), resolution_w,resolution_h)

        org_img_list.append(org_img)  
        con_1_list.append(con_1)
        con_2_list.append(con_2)
        depth_list.append(depth)

    org_imgs = torch.tensor(np.stack(org_img_list, axis=0)).permute(0, 3, 1, 2).float().mean(dim=1,keepdim=True)
    con1_imgs = torch.tensor(np.stack(con_1_list, axis=0)).permute(0, 3, 1, 2).float().mean(dim=1,keepdim=True)
    con2_imgs = torch.tensor(np.stack(con_2_list, axis=0)).permute(0, 3, 1, 2).float().mean(dim=1,keepdim=True)
    depth_imgs = torch.tensor(np.stack(depth_list, axis=0)).permute(0, 3, 1, 2).float().mean(dim=1,keepdim=True)

    org_imgs = (org_imgs / 255.0 - 0.5) * 2.0
    # TODO: condition 是否有更好的数值区间
    con1_imgs = con1_imgs / 255.0
    con2_imgs = con2_imgs / 255.0
    depth_imgs = depth_imgs / 255.0
    # save_videos_grid(videos=org_imgs.unsqueeze(0).permute(0,2,1,3,4), 
    #                  path=f"{savedir}/org.gif", rescale=True)
    # save_videos_grid(videos=con1_imgs.unsqueeze(0).permute(0,2,1,3,4), 
    #                  path=f"{savedir}/con1.gif")
    # save_videos_grid(videos=con2_imgs.unsqueeze(0).permute(0,2,1,3,4), 
    #                  path=f"{savedir}/con2.gif")
    # save_videos_grid(videos=depth_imgs.unsqueeze(0).permute(0,2,1,3,4), 
    #                  path=f"{savedir}/depth.gif")
    cons_list = []
    if con1_wo_texture == True:
        cons_list.append(con1_imgs)
        # print("con1 loaded!")
    if con2_texture == True:
        cons_list.append(con2_imgs)
        # print("con2 loaded!")
    if con3_depth == True:
        cons_list.append(depth_imgs)
        # print("con3 loaded")
    assert len(cons_list) > 0

    adapter_input = torch.cat(cons_list, dim=1)
    assert 64 * adapter_input.shape[1] == cin

    return adapter_input


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"all_lora_all_templates_generation_vis"

    os.makedirs(savedir,exist_ok=True)

    config  = OmegaConf.load(args.config)
    samples = []
    
    sample_idx = 0
    args.W=config["FilmVelvia"].W

    orig_w=args.W
    args.W=math.ceil(args.W/64.0)*64
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:

            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))  # NOTE
        
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
            # 1.1 motion module, 从 tos 上进行读取
            if model_config.motion_module_in_tos == True:
                with smart_open(motion_module, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    motion_module_state_dict = torch.load(buffer, map_location="cpu")
            else:
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
            if adapter_module != "":
                if model_config.adapter_module_in_tos == True:
                    with smart_open(adapter_module, 'rb') as f:
                        buffer = io.BytesIO(f.read())
                        adapter_module_state_dict = torch.load(buffer, map_location="cpu")
                else:
                    adapter_module_state_dict = torch.load(adapter_module, map_location="cpu")
                if "state_dict" in adapter_module_state_dict:
                    adapter_module_state_dict = adapter_module_state_dict["state_dict"]
                    adapter_module_state_dict = {x.replace('module.',''):y for x,y in adapter_module_state_dict.items()}
                from animatediff.models.adapter_module import Adapter
                model_adapter = Adapter(cin=model_config.adapter_cin, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to("cuda")
                missing, unexpected = model_adapter.load_state_dict(adapter_module_state_dict)
                print(f'model_adapter missing {len(missing)}, unexpected {len(unexpected)}')
                lora_id=None if model_config.lora_id=="none" else model_config.lora_id
                adapter_input = get_adapter_input(
                    video_id = model_config.video_id,
                    lora_id = lora_id,
                    start_idx = model_config.start_f_idx,
                    frame_stride = model_config.frame_stride, 
                    video_length = args.L,
                    val_data_prefix = model_config.val_data_prefix,
                    resolution_w = args.W,
                    resolution_h =args.H,
                    con1_wo_texture = model_config.con1_wo_texture,
                    con2_texture = model_config.con2_texture,
                    con3_depth = model_config.con3_depth,
                    cin = model_config.adapter_cin, 
                    deca_replace = model_config.deca_replace,
                    savedir = savedir,
                ).to('cuda')


                adapter_features = model_adapter(adapter_input)
                # adapter_features = model_adapter(torch.concat((frames_ldmks, torch.zeros_like(frames_ldmks)),dim=1))
                # transfer adapter_features to video format (b c f h w)
                adapter_features = [x.permute(1,0,2,3).unsqueeze(0).repeat(2,1,1,1,1) for x in adapter_features]
                print('adapter_features', len(adapter_features), adapter_features[0].shape)
            else:
                adapter_features = None

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
                    print('is_lora:', is_lora)
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
                for seed_idx in range(5):
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
                        adapter_features    = adapter_features,
                    ).videos
                    samples.append(sample)
                    prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                    save_videos_grid(sample, f"{savedir}/{model_config.video_id}/{lora_id}/{sample_idx}-{seed_idx}.gif",resize_shape=(orig_w,args.H))
                    print(f"save to {savedir}/{model_config.video_id}/{lora_id}/{sample_idx}-{seed_idx}.gif")

                sample_idx += 1

    samples = torch.concat(samples)
    # save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    # OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/data00/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0",)
    parser.add_argument("--unet3d_pretrained_model_path", type=str, default='/data00/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0',)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v2.yaml")   # NOTE 
    parser.add_argument("--config",                type=str, default=None)
    parser.add_argument("--all_config_path",type=str,required=True)
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--device", type=int, choices=[0,1,2,3],required=True)
    args = parser.parse_args()
    config_paths=sorted(os.listdir(args.all_config_path))[args.device::4]
    for config_path in config_paths:
        if config_path.endswith("yaml"):
            args.config=os.path.join(args.all_config_path,config_path)
            print("processing:{}".format(args.config))
            main(args)
