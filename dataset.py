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
def load_msgpack_list(prefix, file_path: str):
    file_path = os.path.join(prefix, file_path)
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
                 subsample=None,
                 video_length=16,
                 resolution=[256, 256],
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
                self.data_dir, 'worker*','*.msgpack'
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
                video_meta = load_msgpack_list(self.data_prefix, meta_f)
                # video_meta[0].keys(): ['frames', 'video_file', 'num_frames']
                # assert len(video_meta) ==1
                video_meta = video_meta[0]
                num_frames = video_meta['num_frames']
                
                '''
                frame_metas: dict
                    {frame_idx: 
                        {
                            'caption': "The 30 years old male's emotion is sad", 
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
                for frame_meta in select_frame_metas:
                    frames_captions.append(frame_meta['caption'])
                    img_path = os.path.join(self.data_prefix, frame_meta['img'])
                    frames_imgs_pathes.append(img_path)
                    frames_landmarks_pathes.append(frame_meta['landmarks'])
                worker_info = torch.utils.data.get_worker_info()
                # print('frames_landmarks_pathes', frames_landmarks_pathes)
                # libpng error: bad parameters to zlib                                                                                                                                                          
                # frames_imgs = [cv2.imread(x) for x in frames_imgs_pathes]

                frames_imgs = [np.array(Image.open(os.path.join(self.data_prefix, x)))[...,::-1] for x in frames_imgs_pathes]
                # print('frames_imgs', frames_imgs[0].shape)
                frames_face_infos = [pickle.load(open(os.path.join(self.data_prefix, x),'rb')) for x in frames_landmarks_pathes]
                # print('frames_face_infos', frames_face_infos)

                # 只保留了一个置信度最高的landmark
                # landmarks[0].keys()：['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding']
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
                h_delta = min(int(h-y_max), y_min)
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
                    img_tensor = torch.zeros(new_height, new_width,1)
                    for x,y in landmark:
                        xc = x - crop_x1
                        yc = y - crop_y1
                        scale_x = new_width / (crop_x2-crop_x1)
                        scale_y = new_height / (crop_y2-crop_y1)
                        xr = int(xc * scale_x)
                        yr = int(yc * scale_y)
                        img_tensor[yr,xr] = 1
                    frames_landmarks.append(img_tensor)
                #############################

                frames_imgs = torch.tensor(np.array(frames_imgs)).permute(0, 3, 1, 2).float()
                frames_landmarks = torch.stack(frames_landmarks).permute(0, 3, 1, 2).float()
                
                frames_imgs = F.interpolate(input=frames_imgs[...,crop_y1:crop_y2, crop_x1:crop_x2], size=(new_height, new_width), mode='bilinear', align_corners=False)
                # frames_landmarks = F.interpolate(input=frames_landmarks[...,crop_y1:crop_y2, crop_x1:crop_x2], size=self.resolution, mode='bilinear', align_corners=False)

                frames_imgs = (frames_imgs / 255 - 0.5) * 2
                frames_caption_one = frames_captions[0] + ', then turns into ' + frames_captions[-1].split(" ")[-1]
                # print('frames_caption_one:', frames_caption_one)
                
                if self.is_image:
                    frames_imgs = frames_imgs[0]
                    frames_landmarks = frames_landmarks[0]
                
                sample = dict(pixel_values=frames_imgs, 
                            landmarks=frames_landmarks, 
                            text=frames_caption_one
                            )
                print('sample')
                for x,y in sample.items():
                    try:
                        print(x, y.shape)
                    except:
                        print(x, y)
                        continue
                break
            except:
                f_idx += 1
                continue
        return sample
    def __len__(self):
        return len(self.meta_info_list)
    

if __name__ == '__main__':
    data_prefix = 's3://ljj-sh/Datasets/Videos'
    data_dir = 's3://ljj-sh/Datasets/Videos/videos1600_gen'
    frame_stride = 1
    is_image = True
    save_dir = 'debug'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = S3VideosDataset(data_dir, data_prefix, video_length=16, frame_stride=frame_stride, is_image=is_image)
    # print('dataset size is ', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1,)
    
    for idx, batch in enumerate(dataloader):
        caption = batch["text"][0]
        
        print(batch["pixel_values"].shape, batch["text"])
        print(batch["pixel_values"].max(), batch["pixel_values"].min())
        if is_image:
            image_array = ((batch["pixel_values"][0].permute(1, 2, 0)+1)/2 * 255).numpy().astype(np.uint8)
            ldmks_array = ((batch["landmarks"][0].permute(1, 2, 0)) * 255).numpy().astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{caption}_image.png", image_array)
            cv2.imwrite(f"{save_dir}/{caption}_ldmks.png", ldmks_array)

        else:
            batch["pixel_values"] = batch["pixel_values"] * (1-batch["landmarks"])
            video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
            ldmks_array = ((batch["landmarks"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)

            
            print('caption:', caption)
            with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video.mp4", fps=30) as video_writer:
                for frame in video_array:
                    video_writer.append_data(frame)
            
            with imageio.get_writer(f"{save_dir}/{idx}_{caption}_ldmks.mp4", fps=30) as video_writer:
                for frame in ldmks_array:
                    video_writer.append_data(frame)

            with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video_all.mp4", fps=30) as video_writer:
                for frame,ldmk in zip(video_array,ldmks_array):
                    ldmk = np.repeat(ldmk, repeats=3, axis=2)
                    res = np.hstack((frame,ldmk))
                    video_writer.append_data(res)

        if idx >= 10:
            break
