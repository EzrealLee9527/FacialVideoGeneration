import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

# from megfile import smart_open as open 
import pickle
import pdb

from PIL import Image

from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def get_subdirectories(path):
    subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirectories

def main(args):
    # 创建 DECA 模型
    device = args.device
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)
    savefolder = args.savefolder

    # 读取 LoRA 用户图片
    loradata = datasets.LoraData(args.lora_img_path, iscrop=args.iscrop, face_detector=args.detector,crop_size=224)
    lora_images=[]
    for i in range(len(loradata)):
        lora_images.append(loradata[i]['image'].to(device)[None,...])
    lora_images=torch.cat(lora_images,dim=0)
        
    # print(lora_images.shape)

    with torch.no_grad():
        lora_codedict = deca.encode(lora_images)
    # lora_codedict 中的 keys：['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail']

    # 一个一个处理 refer 视频帧
    video_id_list = get_subdirectories(savefolder)
    
    for video_id in tqdm(video_id_list):

        print("video_id: ", video_id)
        video_dir = os.path.join(savefolder, str(video_id))
        # if os.path.exists(os.path.join(savefolder, str(video_id), args.savename)):
        #     print("already exists, skip")
        #     continue

        frame_id_list = get_subdirectories(video_dir)

        for frame_id in tqdm(frame_id_list):

            # print("frame_id: ", frame_id)

            # 读取 refer_codedict
            refer_codedict_path = os.path.join(savefolder, f"{video_id}/{frame_id}/codedict.pkl")
            with open(refer_codedict_path, 'rb') as f:
                refer_codedict = pickle.load(f)
    
            # transfer: 用 exp_codedict 中的部分潜码代替 id_codedict 中的部分潜码
            transfer_codedict = {}
            transfer_codedict['images'] = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
            transfer_codedict['shape'] = lora_codedict['shape'].to(device)
            transfer_codedict['tex'] = lora_codedict['tex'].to(device)
            transfer_codedict['detail'] = lora_codedict['detail'].to(device)
            transfer_codedict['exp'] = torch.tensor(refer_codedict['exp']).to(device)
            transfer_codedict['pose'] = torch.tensor(refer_codedict['pose']).to(device)
            transfer_codedict['cam'] = torch.tensor(refer_codedict['cam']).to(device)
            transfer_codedict['light'] = torch.tensor(refer_codedict['light']).to(device)

            # save_prefix = os.path.join(savefolder, str(video_id), args.savename, str(frame_id))
            # os.makedirs(save_prefix, exist_ok=True)

            # 进行 render_org 的操作
            tform = torch.tensor(refer_codedict['tform'])
            tform = torch.inverse(tform).transpose(1,2).to(device)
            # 读取 original_image，并且是(c,h,w)的形状
            orig_img_path = refer_codedict_path.replace("codedict.pkl", "orig_inputs.jpg")
            original_image = np.array(Image.open(orig_img_path).convert("RGB"))
            original_image = torch.tensor(original_image.transpose(2,0,1)).float()
            original_image = original_image[None, ...].to(device)
            original_image = torch.zeros_like(original_image)

         
            _, orig_visdict = deca.decode(transfer_codedict, render_orig=True, original_image=original_image, tform=tform)
            orig_visdict['inputs'] = original_image 

            save_prefix = os.path.join(savefolder, str(video_id), str(frame_id),args.savename)
            os.makedirs(save_prefix, exist_ok=True)
            for vis_name in ['rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in orig_visdict.keys():
                    continue
                cv2.imwrite(os.path.join(save_prefix,  'orig_'+vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    # NOTE:更改下面两项!!
    parser.add_argument('-e', '--lora_img_path', default='/data00/fsq/metric/data/benchmark/emma/', type=str, 
                        help='path to expression')
    parser.add_argument('--savename', default='emma1', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')

    parser.add_argument('-s', '--savefolder', default='/data00/Datasets/DECA_examples_v2/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,  
                        help='detector for cropping face, check detectos.py for details' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    print(parser.parse_args())
    # main(parser.parse_args())

    main(parser.parse_args())