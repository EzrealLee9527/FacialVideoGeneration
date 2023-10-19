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
    loradata = datasets.TestData(args.lora_path, iscrop=args.iscrop, face_detector=args.detector)
    lora_images = loradata[0]['image'].to(device)[None,...]
    with torch.no_grad():
        lora_codedict = deca.encode(lora_images)
    # lora_codedict 中的 keys：['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail']

    # 对 lora 进行 decode
    lora_opdict, lora_visdict = deca.decode(lora_codedict)

    # 一个一个处理 refer 视频帧
    video_id_list = get_subdirectories(savefolder)

    for video_id in tqdm(video_id_list):

        print(video_id)
        video_dir = os.path.join(savefolder, str(video_id), "deca")
        if os.path.exists(os.path.join(savefolder, str(video_id), "deca_replace")):
            print("already exists, skip")
            continue

        frame_id_list = get_subdirectories(video_dir)

        for frame_id in tqdm(frame_id_list):

            # 读取 refer_codedict
            refer_codedict_path = os.path.join(savefolder, f"{video_id}/deca/{frame_id}/codedict.pkl")
            with open(refer_codedict_path, 'rb') as f:
                refer_codedict = pickle.load(f)
    
            # 对 refer 进行 decode
            refer_opdict, refer_visdict = deca.decode(refer_codedict)
            # refer_visdict 中的内容：['inputs', 'landmarks2d', 'landmarks3d', 'shape_images', 'shape_detail_images']
    
            # transfer: 用 exp_codedict 中的部分潜码代替 id_codedict 中的部分潜码
            transfer_codedict = {}
            transfer_codedict['images'] = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device='cuda')
            transfer_codedict['shape'] = lora_codedict['shape']
            transfer_codedict['tex'] = lora_codedict['tex']
            transfer_codedict['detail'] = lora_codedict['detail']
            transfer_codedict['exp'] = refer_codedict['exp']
            transfer_codedict['pose'] = refer_codedict['pose']
            transfer_codedict['cam'] = refer_codedict['cam']
            transfer_codedict['light'] = refer_codedict['light']

            transfer_opdict, transfer_visdict = deca.decode(transfer_codedict)
            visdict = transfer_visdict; opdict = transfer_opdict
            save_prefix = os.path.join(savefolder, str(video_id), 'deca_replace', str(frame_id))
            os.makedirs(save_prefix, exist_ok=True)

            if args.saveDepth:
                depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
                visdict['depth_images'] = depth_image
                cv2.imwrite(os.path.join(save_prefix, 'depth.jpg'), util.tensor2image(depth_image[0]))

            if args.saveKpt:
                np.savetxt(os.path.join(save_prefix, 'kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
                np.savetxt(os.path.join(save_prefix, 'kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())

            if args.saveObj:
                deca.save_obj(os.path.join(save_prefix, 'faceMesh.obj'), opdict)

            if args.saveMat:
                opdict = util.dict_tensor2npy(opdict)
                savemat(os.path.join(save_prefix, 'faceMat.mat'), opdict)

            if args.saveImages:
                for vis_name in ['rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                    if vis_name not in visdict.keys():
                        continue
                    image  =util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(os.path.join(save_prefix,  vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')


    parser.add_argument('-e', '--lora_path', default='lora_img/lora_id_38.png', type=str, 
                        help='path to expression')
    # 这里的 savefolder 不再有效了
    parser.add_argument('-s', '--savefolder', default='/data00/Datasets/DECA_examples/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
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
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    print(parser.parse_args())
    # main(parser.parse_args())

    main(parser.parse_args())