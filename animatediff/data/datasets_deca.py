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
# from skimage.transform import resize
import facer

def read_vis_single(vis_img_io, resize_w,resize_h=None):

    # img = imageio.imread("/data00/Datasets/DECA_examples/10040716/1/vis.jpg")
    img = Image.open(vis_img_io).convert("RGB")
    if resize_h is None:
        resize_h=resize_w
    resize_transform = transforms.Resize((resize_h,resize_w))
    img = resize_transform(img)

    return np.array(img)

def read_vis(vis_img_io, resize_w):

    # img = imageio.imread("/data00/Datasets/DECA_examples/10040716/1/vis.jpg")
    img = Image.open(vis_img_io).convert("RGB")

    w = 224
    resize_transform = transforms.Resize((resize_w, resize_w))
    org_img = resize_transform(img.crop((0,0,w,w)))
    con_1 = resize_transform(img.crop((3*w,0,4*w,w)))
    con_2 = resize_transform(img.crop((4*w,0,5*w,w)))
    depth = resize_transform(img.crop((5*w,0,6*w,w)))

    return np.array(org_img), np.array(con_1), np.array(con_2), np.array(depth)

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
                 ldmk_use_gaussian=False,
                 subsample=None,
                 video_length=16,
                 resolution=[224, 224],
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

        # 创建一个 face detector
        self.face_detector = facer.face_detector('retinaface/mobilenet', device="cpu")

    
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
                org_img_list = []
                con_1_list = []
                con_2_list = []
                depth_list = []

                # TODO：筛数据

                # 'img' example: videos1600_gen/worker0/conditions/3402898/frames/761.png
                for frame_meta in select_frame_metas:
                    img_path = frame_meta['img']
                    vis_img_path = os.path.join(self.data_prefix, img_path[:-4].replace("frames", "deca"), "vis.jpg")
                    org_img, con_1, con_2, depth = read_vis((open(vis_img_path, "rb")), self.resolution[1])
                    # imageio.imsave("tmp_test.png", org_img)
                    org_img_for_face_det = torch.tensor(org_img).to(torch.uint8).unsqueeze(0).permute(0, 3, 1, 2)
                    with torch.inference_mode():
                        faces = self.face_detector(org_img_for_face_det)
                        assert faces['image_ids'].numel() == 1, "Image must has exactly one face!"
                    
                    # face_parser = facer.face_parser('farl/lapa/448', device='cuda') # optional "farl/celebm/448"
                    # with torch.inference_mode():
                    #     faces = face_parser(org_img_for_face_det, faces)

                    # np.array uint8, (224,224,3),  0 - 255
                    org_img_list.append(org_img)  
                    con_1_list.append(con_1)
                    con_2_list.append(con_2)
                    depth_list.append(depth)

                    frames_captions.append(frame_meta['caption'])

                org_imgs = torch.tensor(np.stack(org_img_list, axis=0)).permute(0, 3, 1, 2).float()
                con1_imgs = torch.tensor(np.stack(con_1_list, axis=0)).permute(0, 3, 1, 2).float()
                con2_imgs = torch.tensor(np.stack(con_2_list, axis=0)).permute(0, 3, 1, 2).float()
                depth_imgs = torch.tensor(np.stack(depth_list, axis=0)).permute(0, 3, 1, 2).float()
            
                org_imgs = (org_imgs / 255.0 - 0.5) * 2.0
                # TODO: condition 是否有更好的数值区间
                con1_imgs = con1_imgs / 255.0
                con2_imgs = con2_imgs / 255.0
                depth_imgs = depth_imgs / 255.0

                frames_captions = ['The person is ' + x.split(" ")[-1] for x in frames_captions]
                assert(len(frames_captions) > 0)

                if self.is_image:
                    org_imgs = org_imgs[0]
                    con1_imgs = con1_imgs[0]
                    con2_imgs = con2_imgs[0]
                    depth_imgs = depth_imgs[0]

                sample = dict(
                    pixel_values = org_imgs,
                    con1_imgs = con1_imgs,
                    con2_imgs = con2_imgs,
                    depth_imgs = depth_imgs,
                    texts = frames_captions,
                )
                
                # print("Successfully Get One Data")
                break
            except Exception as e:
                f_idx += 1
                # print("Error: ", e)
                continue
        
        return sample
    def __len__(self):
        return len(self.meta_info_list)

def gaussian(x, y, sigma):
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    return np.exp(exponent) / (2 * np.pi * sigma**2)

def generate_gaussian_response(image_shape, landmarks, sigma=3):
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in landmarks:
        if 0 <= x < width and 0 <= y < height:
            for i in range(-sigma, sigma+1):
                for j in range(-sigma, sigma+1):
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < width and 0 <= new_y < height:
                        heatmap[new_y, new_x] += gaussian(i, j, sigma)                        
    heatmap /= np.max(heatmap)
    heatmap = heatmap[:,:,np.newaxis]
    return heatmap 

if __name__ == '__main__':
    vis_img_path = "s3://ljj-sh/Datasets/Videos/videos1600_gen/worker2/conditions/4943909/deca/5/vis.jpg"
    a = read_vis((open(vis_img_path, "rb")), 224)

    data_prefix = 's3://ljj-sh/Datasets/Videos'
    # data_dir = 's3://ljj-sh/Datasets/Videos/videos1600_gen'
    data_dir = 's3://ljj-sh/Datasets/Videos/msgpacks/videos_231002/'
    frame_stride = 1
    is_image = False
    video_length = 16
    save_dir = f'debug{video_length}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = S3VideosDataset(data_dir, data_prefix, ldmk_use_gaussian=True, video_length=video_length, frame_stride=frame_stride, is_image=is_image)
    # print('dataset size is ', len(dataset))

    a = dataset[0]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1,)
    
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, batch["texts"])  # (2, 16, 3, 224, 224)
        print(batch["pixel_values"].max(), batch["pixel_values"].min())
        caption = batch["texts"][0][0].replace(' ', '_')

        batch["pixel_values"] = batch["pixel_values"] 
        video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
        cons1_array = ((batch["con1_imgs"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
        cons2_array = ((batch["con2_imgs"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
        depth_array = ((batch["depth_imgs"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)

        print('caption:', caption)
        print('video_array', video_array.shape, video_array.max(), video_array.min())
        print('cons1_array', cons1_array.shape, cons1_array.max(), cons1_array.min())
        print('cons2_array', cons2_array.shape, cons2_array.max(), cons2_array.min())
        print('depth_array', depth_array.shape, depth_array.max(), depth_array.min())

        with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video.mp4", fps=30) as video_writer:
            for frame in video_array:
                video_writer.append_data(frame)
        
        with imageio.get_writer(f"{save_dir}/{idx}_{caption}_cons1_frames{video_length}.mp4", fps=30) as video_writer:
            for frame in cons1_array:
                video_writer.append_data(frame)
            
        with imageio.get_writer(f"{save_dir}/{idx}_{caption}_cons2_frames{video_length}.mp4", fps=30) as video_writer:
            for frame in cons2_array:
                video_writer.append_data(frame)

        with imageio.get_writer(f"{save_dir}/{idx}_{caption}_depth_frames{video_length}.mp4", fps=30) as video_writer:
            for frame in depth_array:
                video_writer.append_data(frame)

        with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video_all.mp4", fps=30) as video_writer:
            for frame,cons1,cons2,depth in zip(video_array,cons1_array, cons2_array, depth_array):
                res = np.hstack((frame, cons1, cons2, depth))
                video_writer.append_data(res)

        if idx >= 200:
            break
