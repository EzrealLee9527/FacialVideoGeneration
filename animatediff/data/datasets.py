from torch.utils.data import IterableDataset
import io
import boto3
import os
import random
import bisect
import pandas as pd
import torch.nn.functional as F
import omegaconf
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import imageio
import numpy as np
import cv2
import json
import sys
from PIL import Image, ImageDraw
# from animatediff.data.data_utils import check_file_exists, load_csv, load_json, load_txt, get_video_reader
# from data_utils import check_file_exists, load_csv, load_json, load_txt, get_video_reader

from glob import glob
import tarfile
from megfile import smart_open as open
import megfile
import msgpack
from megfile import smart_glob
import pickle

# 放到data_utils里
def load_msgpack_list(file_path: str):
    loaded_data = []
    with open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f,strict_map_key = False)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data


# @lru_cache(maxsize=128)
def load_tar(p):
    return tarfile.open(fileobj=open(p, 'rb'))


def load_img_from_tar(img_path):
    tar_fname,img_fname = img_path.rsplit("/",1)
    tar_obj = load_tar(tar_fname)
    img = Image.open(tar_obj.extractfile(img_fname)).convert("RGB")
    return np.array(img)

def read_remote_img(p):
    with open(p, 'rb') as rf:
        return Image.open(rf).convert("RGB")

def gen_landmark_control_input(img_tensor, landmarks):
    cols = torch.tensor([int(y) for x,y in landmarks])
    rows = torch.tensor([int(x) for x,y in landmarks])
    img_tensor = img_tensor.index_put_(indices=(cols, rows), values=torch.ones(106))
    return img_tensor.unsqueeze(-1)

class S3VideosDataset(Dataset):   
    def __init__(self,
                 data_dir,
                 data_prefix,
                 local_data_prefix,
                 use_faceparsing=False,
                 ldmk_use_gaussian=False,
                 subsample=None,
                 video_length=16,
                 resolution=[512, 512],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 ):
        self.data_prefix = data_prefix
        # self.data_dir = os.path.join(self.data_prefix, data_dir)
        self.data_dir = data_dir
        
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(
            resolution, int) else resolution
        self.frame_stride = frame_stride
        self.fps_max = fps_max
        self.load_raw_resolution = load_raw_resolution
        self.fs_probs = fs_probs
        self.trigger_word = trigger_word
        self.dataname = dataname
        self.is_image = is_image
        self.info = self._get_file_info()
        self.start = self.info['start']
        self.end = self.info['end']
        self.ldmk_use_gaussian = ldmk_use_gaussian
        self.use_faceparsing = use_faceparsing
        self.local_data_prefix = local_data_prefix
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "resize_center_crop":
                assert (self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(resolution),
                    transforms.CenterCropVideo(resolution),
                ])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

        self.fps_schedule = fps_schedule
        self.bs_per_gpu = bs_per_gpu
        if self.fps_schedule is not None:
            assert (self.bs_per_gpu is not None)
            self.counter = 0
            self.stage_idx = 0
    
    def _get_file_info(self):
        self.meta_info_list = smart_glob(
            os.path.join(
                self.data_dir, '*.msgpack'
            )
        )
        print(f'Detect {len(self.meta_info_list)} meta files')
        info = {
            "start": 0,
            "end": len(self.meta_info_list),
        }
        return info
    
    def __getitem__(self, f_idx):
        while True:
            f_idx = f_idx % len(self.meta_info_list)
            try:
                meta_f = self.meta_info_list[f_idx]
                video_meta = load_msgpack_list(meta_f)
                # video_meta[0].keys(): ['frames', 'video_file', 'num_frames']
                # assert len(video_meta) ==1
                video_meta = video_meta[0]
                num_frames = video_meta['num_frames']
                
                '''
                frame_metas: dict
                    {frame_idx: 
                        {
                            'caption': "The 30 years old female's emotion is sad, then turns into sad", 
                            'img': 'videos1600_gen/worker0/conditions/10040716/frames/0.png', 
                            'landmarks': 'videos1600_gen/worker0/conditions/10040716/landmarks/0.pkl'
                        }
                    }
                '''
                frame_metas = video_meta['frames']

                # sample strided frames
                frame_stride = self.frame_stride
                all_frames = list(range(0, num_frames, frame_stride))
                if len(all_frames) < self.video_length:  # recal a max fs
                    frame_stride = num_frames // self.video_length
                    assert (frame_stride != 0)
                    all_frames = list(range(0, num_frames, frame_stride))

                # select a random clip
                rand_idx = random.randint(0, len(all_frames) - self.video_length)
                frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
                # print('frame_indices', frame_indices)
                select_frame_metas = [frame_metas[i] for i in frame_indices]
                frames_captions = []
                frames_imgs_pathes = []
                frames_landmarks_pathes = []
                frames_faceparsing_pathes = []
                # 'img' example: videos1600_gen/worker0/conditions/3402898/frames/761.png
                # 'landmarks' example: videos1600_gen/worker0/conditions/3402898/landmarks/772.pkl
                # fsq gen fp: /data00/Datasets/Videos/videos4000_gen/worker2/conditions/6033644/labels/51.png
                # self.local_data_prefix = '/data00/'
                
                for frame_meta in select_frame_metas:
                    frames_captions.append(frame_meta['caption'])
                    frames_imgs_pathes.append(frame_meta['img'].replace(self.local_data_prefix, self.data_prefix))
                    frames_landmarks_pathes.append(frame_meta['landmarks'].replace(self.local_data_prefix, self.data_prefix))
                    if self.use_faceparsing:
                        frames_faceparsing_pathes.append(frame_meta['faceparsing'].replace(self.local_data_prefix, self.data_prefix))
                worker_info = torch.utils.data.get_worker_info()
                # print('frames_landmarks_pathes', frames_landmarks_pathes)
                # libpng error: bad parameters to zlib                                                                                                                                                          
                # frames_imgs = [cv2.imread(x) for x in frames_imgs_pathes]

                # RGB mode
                frames_imgs = [np.array(Image.open(open(x,'rb'))) for x in frames_imgs_pathes]
                '''
                face parsing:
                0-10 分别代表： ['background', 'face', 'rb', 'lb', 're', 'le', 'nose', 'ulip', 'imouth', 'llip', 'hair']                
                1. 数值为10的头发舍掉
                2. 多张脸重叠，有可能导致 数值>10，也应该舍掉
                '''
                if self.use_faceparsing:
                    faceparsing_imgs = []
                    face_masks = []
                    for x in frames_faceparsing_pathes:
                        faceparsing_img = np.array(Image.open(open(x,'rb')))[:,:,:1]
                        face_masks.append(faceparsing_img>0)
                        faceparsing_img[faceparsing_img>=10] = 0
                        faceparsing_imgs.append(faceparsing_img)                

                frames_face_infos = [pickle.load(open(x,'rb')) for x in frames_landmarks_pathes]
                # print('frames_face_infos', frames_face_infos)

                # 只保留了一个置信度最高的landmark
                # frames_face_infos[0].keys()：['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding']
                landmarks = [x[0]['landmark_2d_106'] for x in frames_face_infos]
                h, w, c = frames_imgs[0].shape
                # img_tensor=torch.zeros((h,w), dtype=torch.float)
                # frames_landmarks = [gen_landmark_control_input(img_tensor, x) for x in landmarks]

                # crop and resize according face
                x_min, y_min, x_max, y_max = frames_face_infos[0][0]['bbox']
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(w, int(x_max))
                y_max = min(h, int(y_max))
                h_delta = min(int(h-y_max), y_min, (y_max-y_min)//2)
                crop_y1 = max(y_min - h_delta, 0)
                crop_y2 = min(y_max + h_delta, h)
                h = crop_y2 -crop_y1
                w_delta = max(0, (h - (x_max-x_min)) / 2)
                crop_x1 = max(int(x_min - w_delta),0)
                crop_x2 = min(int(x_max + w_delta),w)
                #############################
                frames_landmarks = []
                new_height, new_width = self.resolution 
                  
                for landmark in landmarks:
                    new_landmarks = [] 
                    img_tensor = torch.zeros(new_height, new_width,1)
                    for x,y in landmark:
                        xc = x - crop_x1
                        yc = y - crop_y1
                        scale_x = new_width / (crop_x2-crop_x1)
                        scale_y = new_height / (crop_y2-crop_y1)
                        xr = int(xc * scale_x)
                        yr = int(yc * scale_y)
                        img_tensor[yr,xr] = 1
                        new_landmarks.append((xr,yr))
                        
                    if self.ldmk_use_gaussian:
                        gaussian_response = generate_gaussian_response([new_height, new_width], new_landmarks, sigma=1)
                        frames_landmarks.append(gaussian_response) 
                    else:           
                        frames_landmarks.append(img_tensor)
                #############################
                if self.ldmk_use_gaussian:
                    frames_landmarks = torch.tensor(np.array(frames_landmarks)).permute(0, 3, 1, 2).float()   
                else:
                    frames_landmarks = torch.stack(frames_landmarks).permute(0, 3, 1, 2).float()

                size = (new_height, new_width)
                frames_imgs = interpolate(frames_imgs, crop_y1, crop_y2, crop_x1, crop_x2, size)
                frames_imgs = (frames_imgs / 255 - 0.5) * 2
                if self.use_faceparsing:
                    faceparsing_imgs = interpolate(faceparsing_imgs, crop_y1, crop_y2, crop_x1, crop_x2, size)
                    faceparsing_imgs = faceparsing_imgs / 11.0
                    face_masks = interpolate(face_masks, crop_y1, crop_y2, crop_x1, crop_x2, size)                

                frames_caption_one = 'The person turns ' +  frames_captions[0].split(" ")[-1] + ' into ' + frames_captions[-1].split(" ")[-1]
                # print('frames_caption_one:', frames_caption_one)
                
                if self.is_image:
                    frames_imgs = frames_imgs[0]
                    frames_landmarks = frames_landmarks[0]
                    if self.use_faceparsing:
                        faceparsing_imgs = faceparsing_imgs[0]
                        face_masks = face_masks[0]
                    
                frames_captions = ['The person is ' + x.split(" ")[-1] for x in frames_captions]
                assert(len(frames_captions) > 0)
                if self.use_faceparsing:
                    sample = dict(pixel_values=frames_imgs, 
                                landmarks=frames_landmarks, 
                                texts=frames_captions,
                                face_parsings = faceparsing_imgs,
                                face_masks = face_masks,
                                )
                else:
                    sample = dict(pixel_values=frames_imgs, 
                                landmarks=frames_landmarks, 
                                texts=frames_captions,
                                )
                
                # print('debug sample')
                # for x,y in sample.items():
                #     try:
                #         print(x, y.shape)
                #     except:
                #         print(x, len(y))
                #         # print(x, y)
                #         continue
                    
                break
            except:
                f_idx += 1
                continue
        
        return sample
    
    def __len__(self):
        return len(self.meta_info_list)

def interpolate(data, crop_y1, crop_y2, crop_x1, crop_x2, size):
    data = torch.tensor(np.array(data)).permute(0, 3, 1, 2).float()
    data = F.interpolate(input=data[...,crop_y1:crop_y2, crop_x1:crop_x2], size=size, mode='bilinear', align_corners=False)
    return data

def gaussian(x, y, sigma):
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    return np.exp(exponent) / (2 * np.pi * sigma**2)

def generate_gaussian_response(image_shape, landmarks, sigma=3):
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            for i in range(-sigma, sigma+1):
                for j in range(-sigma, sigma+1):
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < width and 0 <= new_y < height:
                        heatmap[new_y, new_x] += gaussian(i, j, sigma)                        
    
    heatmap[np.isnan(heatmap)] = 0
    max_value = np.max(heatmap)
    if max_value != 0:
        heatmap /= max_value
    heatmap = heatmap[:,:,np.newaxis]
    return heatmap 

if __name__ == '__main__':
    local_data_prefix = 'videos1600_gen/'
    data_prefix = 's3://ljj-sh/Datasets/Videos/videos1600_gen/'
    data_dir = 's3://ljj-sh/Datasets/Videos/msgpacks/videos_231002/'

    # data_prefix = 's3://ljj-sh/'
    # local_data_prefix = '/data00/'
    # data_dir = 's3://ljj-sh/Datasets/Videos/msgpacks/videos4000_gen_labels/'

    ldmk_use_gaussian = True
    frame_stride = 1
    is_image = False
    video_length = 16
    use_faceparsing = False
    save_dir = f'debug{video_length}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = S3VideosDataset(data_dir, data_prefix, local_data_prefix, use_faceparsing=use_faceparsing, ldmk_use_gaussian=True, video_length=video_length, frame_stride=frame_stride, is_image=is_image)
    print('dataset size is ', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1,)
    
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, batch["texts"])
        print(batch["pixel_values"].max(), batch["pixel_values"].min())
        caption = batch["texts"][0][0].replace(' ', '_')
        if is_image:
            image_array = ((batch["pixel_values"][0].permute(1, 2, 0)+1)/2 * 255).numpy().astype(np.uint8)[...,::-1]
            ldmks_array = ((batch["landmarks"][0].permute(1, 2, 0)) * 255).numpy().astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{caption}_image.png", image_array)
            cv2.imwrite(f"{save_dir}/{caption}_ldmks.png", ldmks_array)
        else:
            # batch["pixel_values"] = batch["pixel_values"] * (1-batch["landmarks"])
            video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
            ldmks_array = ((batch["landmarks"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
            if use_faceparsing:
                face_parsings_array = ((batch["face_parsings"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
                masks_array = ((batch["face_masks"][0].permute(0, 2, 3, 1))).numpy().astype(np.uint8)
            print('caption:', caption)
            print('video_array', video_array.shape, video_array.max(), video_array.min())
            print('ldmks_array', ldmks_array.shape, ldmks_array.max(), ldmks_array.min())

            with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video.mp4", fps=30) as video_writer:
                for frame in video_array:
                    video_writer.append_data(frame)
            
            with imageio.get_writer(f"{save_dir}/{idx}_{caption}_ldmks_frames{video_length}.mp4", fps=30) as video_writer:
                for frame in ldmks_array:
                    video_writer.append_data(frame)

            if use_faceparsing:
                with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video_all.mp4", fps=30) as video_writer:
                    for frame,ldmk,faceparsing,mask in zip(video_array,ldmks_array,face_parsings_array,masks_array):
                        ldmk = np.repeat(ldmk, repeats=3, axis=2)
                        faceparsing = np.repeat(faceparsing, repeats=3, axis=2)
                        mask = np.repeat(mask, repeats=3, axis=2)
                        frame[mask==0] = 0
                        res = np.hstack((frame,ldmk,faceparsing))
                        video_writer.append_data(res)    
            else:
                with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video_all.mp4", fps=30) as video_writer:
                    for frame,ldmk in zip(video_array,ldmks_array):
                        ldmk = np.repeat(ldmk, repeats=3, axis=2)
                        res = np.hstack((frame,ldmk))
                        video_writer.append_data(res)    

        if idx >= 100:
            break
