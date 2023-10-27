# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from megfile import smart_glob, smart_open
import msgpack
import pickle
import io
from megfile import smart_glob as glob
from megfile import smart_open as open
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
import scipy.io
from decalib.datasets import detectors
import re
import moviepy.editor

'''
opdict包含以下键值对：

'verts': FLAME模型生成的原始顶点。
'trans_verts': 投影后的顶点，表示3D人脸模型在2D平面上的投影。
'landmarks2d': 二维人脸关键点。
'landmarks3d': 三维人脸关键点，在投影到2D空间后的坐标。
'landmarks3d_world': 三维人脸关键点在世界坐标系下的坐标。
代码首先使用FLAME模型根据输入参数解码顶点、2D关键点和3D关键点。接着，如果模型使用了纹理，则计算albedo，否则将其设为全零。之后利用util.batch_orth_proj()方法将3D关键点、顶点等进行正交投影，并将投影结果存储在opdict中。

在## rendering部分，代码通过调用render()方法来渲染给定参数的3D人脸，并将结果存储在opdict中。同时，根据条件，也会处理其他相关参数，如uv_texture、normals、uv_detail_normals和displacement_map等，并将它们添加到输出字典opdict中。

opdict数据包括3D人脸形状、纹理、投影坐标、关键点以及渲染结果等。
'''
def get_number(s):
    return int(re.search(r'\d+(?=.png)', s).group())


def dump_img(arr, f):
    with smart_open(f, 'wb') as wf:
        postfix = f.split(".")[-1].lower()
        content = cv2.imencode("."+postfix, arr)[1].tobytes()
        wf.write(content)

def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

def main(args):
    if args.rasterizer_type != 'standard':
        args.render_orig = False
    args.render_orig = True
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    # testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # files = smart_glob('s3://ljj-sh/Datasets/Videos/msgpacks/videos_231002/*.msgpack')
    
    '''
    s3://ljj-sh/Datasets/Videos/videos1600_gen/worker0/conditions/4107560/landmarks/
    s3://ljj-sh/Datasets/Videos/videos1600_gen/worker0/*.msgpack

    '''
    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    


    # imagepath_list = []
    # testpath = args.inputpath
    iscrop=True
    crop_size=224
    resolution_inp = crop_size
    scale=1.25
    face_detector='fan'
    face_detector = detectors.FAN()
    # sample_step=10
    templates_path=args.template_path
    bbox=None
    
    # s3_dirs = glob(testpath+'/*')
    # for dir_i in tqdm(range(len(s3_dirs))):
    #     s3_dir = s3_dirs[dir_i]
    for video_template_path in os.listdir(templates_path):
        vid_path=os.path.join(templates_path,video_template_path)
        print("processing:{}".format(vid_path))
        if vid_path.endswith("gif"):
            video=moviepy.editor.VideoFileClip(vid_path)
            video=video.resize(height=512)
        images=video.iter_frames()
        # debug
        if not os.path.exists(os.path.join(savefolder,video_template_path[:-4])):
                os.makedirs(os.path.join(savefolder,video_template_path[:-4]))
        for i,image in enumerate(images):
            savefolder=args.savefolder
            print(image.shape)
            pre_crop = False
            h, w, _ = image.shape
            if iscrop:
                # print('bbox', bbox)
                if bbox is not None and len(bbox) >= 4:
                    left = int(bbox[0])
                    right=int(bbox[2])
                    top = int(bbox[1])
                    bottom=int(bbox[3])
                    # left = bbox[2]; right=bbox[0]
                    # top = bbox[1]; bottom=bbox[3]
                    # print('bbox', left, right, top, bottom)
                    face_w = right - left
                    face_h = bottom - top
                    face_crop_left = max(0, int(left-face_w//2))
                    face_crop_right = min(w-1, int(right+face_w//2))
                    face_crop_top = max(0, int(top-face_h//2))
                    face_crop_bottom = min(h-1, int(bottom+face_h//2))
                    
                    crop_image = image[face_crop_top:face_crop_bottom,face_crop_left:face_crop_right,:]
                    pre_crop = True
                else:
                    crop_image = image

                try:
                    print('crop_image', crop_image.shape)
                    bbox, bbox_type = face_detector.run(crop_image)
                    # print('after face_detector bbox', bbox)
                    # print(bbox[2]-bbox[0],bbox[3]-bbox[1])
                except:
                    # print('no landmark detected! continue')
                    bbox=None
                    continue 
            # 计算相对于原图的bbox
                if pre_crop and bbox is not None and len(bbox) >= 4:
                    bbox[0] = bbox[0] + face_crop_left 
                    bbox[1] = bbox[1] + face_crop_top 
                    bbox[2] = bbox[2] + face_crop_left 
                    bbox[3] = bbox[3] + face_crop_top 
                    
                if len(bbox) < 4:
                    print('no face detected! continue')
                    continue
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = int(bbox[0])
                    right=int(bbox[2])
                    top = int(bbox[1])
                    bottom=int(bbox[3])
                old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                size = int(old_size*scale)
                src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            else:
                src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
            if not os.path.exists(os.path.join(savefolder,video_template_path[:-4],str(i))):
                os.makedirs(os.path.join(savefolder,video_template_path[:-4],str(i)))
            dump_dir=os.path.join(savefolder,video_template_path[:-4],str(i))
            dump_img(image[:,:,::-1], os.path.join(dump_dir, 'orig_inputs.jpg'))

            savefolder=dump_dir
            DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            
            image = image/255.

            dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
            dst_image = dst_image.transpose(2,0,1)

            

            
            original_image = torch.tensor(image.transpose(2,0,1)).float()
            image = torch.tensor(dst_image).float()
            tform = torch.tensor(tform.params).float()            
            
            # save_codedict_path = imagepath.replace('frames', 'deca_codedict').replace('.png','.pkl')
            images = torch.tensor(dst_image).float().to(device)[None,...]
            # print('images', images.shape)
            name=""
            with torch.no_grad():
                # dict_keys(['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail'])
                print("encode image shape:{}".format(images.shape))
                codedict = deca.encode(images)
                codedict_np = {x:y.cpu().numpy() for x,y in codedict.items()}
                codedict_np['bbox'] = np.array(bbox)[None,...]
                codedict_np['tform'] = np.array(tform)[None,...]
                del codedict_np['images']
                # for x,y in codedict_np.items():
                #     print(x,y.shape)
                # print('\n')
                save_codedict_path = os.path.join(savefolder, name+'codedict.pkl')
                # print('save_codedict_path', save_codedict_path)
                with smart_open(save_codedict_path, 'wb') as file:
                    pickle.dump(codedict_np, file)
                opdict, visdict = deca.decode(codedict) #tensor
                # import pdb;pdb.set_trace()
                if args.render_orig:
                    tform = tform[None, ...]
                    tform = torch.inverse(tform).transpose(1,2).to(device)
                    original_image = original_image[None, ...].to(device)
                    original_image = torch.zeros_like(original_image)
                    _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                    orig_visdict['inputs'] = original_image            

            # if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            # os.makedirs(os.path.join(savefolder, name), exist_ok=True)
            # -- save results
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            save_depth_path = os.path.join(savefolder, name+'depth.jpg')
            dump_img(util.tensor2image(depth_image[0]),save_depth_path)
            
            save_kpt2d_path = os.path.join(savefolder,  name+'kpt2d.txt')
            with smart_open(save_kpt2d_path, 'wb') as file:
                np.savetxt(file, opdict['landmarks2d'][0].cpu().numpy())
            
            save_kpt3d_path = os.path.join(savefolder,name+'kpt3d.txt')
            with smart_open(save_kpt3d_path, 'wb') as file:
                np.savetxt(file, opdict['landmarks3d'][0].cpu().numpy())
            # if args.saveObj:
            #     deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
            opdict = util.dict_tensor2npy(opdict)
            save_opdict_path = os.path.join(savefolder, name+'opdict.mat')
            buffer = io.BytesIO()
            savemat(buffer, opdict)
            with smart_open(save_opdict_path, "wb") as file:
                buffer.seek(0)
                file.write(buffer.read())

            save_vis_path = os.path.join(savefolder, name+'vis.jpg')
            dump_img(deca.visualize(visdict),save_vis_path)

            if args.render_orig:
                dump_img(deca.visualize(orig_visdict), os.path.join(savefolder, name+'vis_original_size.jpg'))

            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                dump_img(image, os.path.join(savefolder, name + vis_name +'.jpg'))
                if args.render_orig:
                    if vis_name == 'inputs':
                        continue
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    dump_img(image, os.path.join(savefolder, name + 'orig_' +  '' + vis_name +'.jpg'))
        print(f'-- please check the results in {dump_dir}')
            

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    # parser.add_argument('--worker', default='worker0', type=str)
    # parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
    #                     help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument("--template_path",required=True,type=str)
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())