import os
import random
import bisect

import pandas as pd

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
sys.path.append('/work00/AnimateDiff-adapter')
from animatediff.data.control_signals.landmarks import get_landmarks
from PIL import Image, ImageDraw


class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """

    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 ):
        self.meta_path = meta_path
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

        self._load_metadata()
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

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path)
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        if self.dataname == "loradata":
            rel_video_fp = str(sample['videoid']) + '.mp4'
            full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        else:
            rel_video_fp = os.path.join(
                sample['page_dir'], str(sample['videoid']) + '.mp4')
            full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        # nstage=len_fps_schedule + 1
        assert (len(frame_strides) == len(schedule) + 1)
        global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(
                f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride

    def get_fs_based_on_probs(self, frame_strides, probs):
        assert (len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(
                    self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert (isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_fp = self._get_video_path(sample)
            caption = sample['caption']+self.trigger_word

            # make reader
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(
                        0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(
                        f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            # sample strided frames
            all_frames = list(range(0, len(video_reader), frame_stride))
            if len(all_frames) < self.video_length:  # recal a max fs
                frame_stride = len(video_reader) // self.video_length
                assert (frame_stride != 0)
                all_frames = list(range(0, len(video_reader), frame_stride))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert (
            frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(
            0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] ==
                    self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        # frames = frames / 255

        # fps_ori = video_reader.get_avg_fps()
        # fps_clip = fps_ori // frame_stride
        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        # data = {'video': frames, 'caption': caption, 'path': video_path,
        #         'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        
        
        if self.is_image:
            frames = frames[0]
        sample = dict(pixel_values=frames, text=caption)
        return sample
    
    def __len__(self):
        return len(self.metadata)
    

class SmileGif(Dataset):
    """
    SmileGif Dataset.
    Assumes SmileGif data is structured as follows.
    SmileGif/
        gifs/
            1.gif
            2.gif
            ...
    """

    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=8,
                 resolution=[256, 256],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 ):
        self.meta_path = meta_path
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

        self._load_metadata()
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

    def _load_metadata(self):
        with open(self.meta_path, 'r') as txt_file:
            lines = txt_file.readlines()
            metadata = [x.strip() for x in lines]
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        # metadata['caption'] = metadata['name']
        # del metadata['name']
        self.metadata = metadata
        # self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = str(sample) #+ '.gif'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        # nstage=len_fps_schedule + 1
        assert (len(frame_strides) == len(schedule) + 1)
        global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(
                f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride

    def get_fs_based_on_probs(self, frame_strides, probs):
        assert (len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(
                    self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert (isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path, rel_fp = self._get_video_path(sample)
            # caption = sample['caption']+self.trigger_word
            caption = 'a person is smiling'
            
            
            # make reader
            try:                
                if self.load_raw_resolution:
                    # video_reader = VideoReader(video_path, ctx=cpu(0))
                    video_reader = imageio.mimread(video_path)

                else:
                    json_path = video_path.replace('gifs', 'landmarks').replace('.gif','.json')
                    with open(json_path) as f:
                        info = json.load(f)
                    # video_reader = VideoReader(video_path, ctx=cpu(
                    #     0), width=self.resolution[1], height=self.resolution[0])
                    reader = imageio.mimread(video_path,memtest=False)
                    width, height =  reader[0].shape[:2]
                    # reader = imageio.get_reader(gif_input_path)
                    # width, height =  reader.get_meta_data()['source_size']
                    new_width, new_height = self.resolution

                    frames_ldmks = []
                    frames = []
                    for f_idx in range(len(reader)):
                        frame = reader[f_idx]
                        width, height =  frame.shape[:2]
                        ldmks = info['frames'][f_idx]['faces'][-1]['landmarks']
                        x_list, y_list = [x for x,y in ldmks], [y for x,y in ldmks]
                        x_min,x_max = int(min(x_list)), int(max(x_list))
                        y_min,y_max = int(min(y_list)), int(max(y_list))
                        h_delta = min(int(height-y_max), y_min)
                        crop_y1 = y_min - h_delta
                        crop_y2 = y_max + h_delta
                        h = crop_y2 -crop_y1
                        w_delta = (h - (x_max-x_min)) / 2
                        crop_x1 = int(x_min - w_delta)
                        crop_x2 = int(x_max + w_delta)

                        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                        resized_frame = cv2.resize(cropped_frame, (new_height, new_width))
                        
                        frame_ldmks = np.zeros((new_height, new_width,1))
                        for x,y in ldmks:
                            xc = x - crop_x1
                            yc = y - crop_y1
                            scale_x = new_width / (crop_x2-crop_x1)
                            scale_y = new_height / (crop_y2-crop_y1)
                            xr = int(xc * scale_x)
                            yr = int(yc * scale_y)
                            frame_ldmks[yr,xr] = 1
                        frames_ldmks.append(frame_ldmks)
                        frames.append(resized_frame) 
                if len(frames) < self.video_length or len(reader[0].shape) < 3:
                    print(
                        f"video length ({len(frames)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                # print(f"Load video failed! path = {video_path}")
                continue

            # # sample strided frames
            # all_frames = list(range(0, len(frames), frame_stride))
            # if len(all_frames) < self.video_length:  # recal a max fs
            #     frame_stride = len(frames) // self.video_length
            #     assert (frame_stride != 0)
            #     all_frames = list(range(0, len(frames), frame_stride))

            # select a random clip
            rand_idx = random.randint(0, len(frames) - self.video_length + 1)
            try:
                # frames = video_reader.get_batch(frame_indices)
                frames = frames[rand_idx:rand_idx+self.video_length]
                frames_ldmks = frames_ldmks[rand_idx:rand_idx+self.video_length]

                assert (
                    len(frames) == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
                frames = torch.tensor(np.array(frames)).permute(
                    0, 3, 1, 2).float()[:,:3,:,:]  # [t,h,w,c] -> [t,c,h,w]
                frames_ldmks = torch.tensor(np.array(frames_ldmks)).permute(
                    0, 3, 1, 2).float() # [t,h,w,c] -> [t,c,h,w]
        
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        
        # print('frames', frames.shape, frames_ldmks.shape)
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] ==
                    self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        # frames = frames / 255

        # fps_ori = video_reader.get_avg_fps()
        # fps_clip = fps_ori // frame_stride
        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        # data = {'video': frames, 'caption': caption, 'path': video_path,
        #         'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        
        
        if self.is_image:
            frames = frames[0]
        sample = dict(pixel_values=frames, landmarks=frames_ldmks, text=caption, rel_fp=rel_fp)
        return sample
    
    def __len__(self):
        return len(self.metadata)


wrong_list = []
missing_list = []

def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def drawLandmark_multiple_nobox(img, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    # print('ddddbug', img.shape, (bbox.left, bbox.top), (bbox.right, bbox.bottom))
    img = img.copy()
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

class EmotionsGif_controlnet(Dataset):
    """
    EmotionsGif Dataset.
    Assumes EmotionsGif data is structured as follows.
    collected_emotions_gif/
        Frown/
            1.gif
            2.gif
        Gape/
            1.gif
            2.gif
        girl_smile_gif/
            asia_girl_smile_1.gif
            asia_girl_smile_2.gif
            ...
    """

    def __init__(self,
                 data_dir,
                 emotions_type,
                 meta_path=None,
                 subsample=None,
                 video_length=8,
                 resolution=[512, 512],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 face_parsing_path=None,           # NOTE
                 ):
        self.meta_path = meta_path
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
        self.face_parsing_path = face_parsing_path # NOTE
        
        # self.all_emotions_type = ["all", "frown", "gape", "girl_smile_gif", "Pout", "Raise_eyebrows", "Roll_eyes", "smile", "wink"]
        self.all_emotions_type = os.listdir(self.data_dir)
        assert (emotions_type == "all" or emotions_type in self.all_emotions_type)
        self.emotions_type = emotions_type

        if meta_path != None:
            self._load_metadata()
        else:
            if emotions_type != "all":
                self.metadata = [emotions_type + '/' + gif_file for gif_file in os.listdir(os.path.join(self.data_dir, emotions_type)) if gif_file.endswith(".gif")]
            else:
                self.metadata = []
                for emo_type in self.all_emotions_type:
                    self.metadata.extend([emo_type + '/' + gif_file for gif_file in os.listdir(os.path.join(self.data_dir, emo_type)) if gif_file.endswith(".gif")])
        
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

    def _load_metadata(self):
        with open(self.meta_path, 'r') as txt_file:
            lines = txt_file.readlines()
            metadata = [x.strip() for x in lines]
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        # metadata['caption'] = metadata['name']
        # del metadata['name']
        self.metadata = metadata  # 就是一个列表
        # self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = str(sample) #+ '.gif'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        # nstage=len_fps_schedule + 1
        assert (len(frame_strides) == len(schedule) + 1)
        global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(
                f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride

    def get_fs_based_on_probs(self, frame_strides, probs):
        assert (len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(
                    self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert (isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            # video_path, rel_fp = self._get_video_path(sample)
            # caption = sample['caption']+self.trigger_word
            
            video_path = os.path.join(self.data_dir, sample)
            rel_fp = sample

            if self.emotions_type == "all":
                emo_type = video_path.split('/')[-2]
            else:
                emo_type = self.emotions_type

            # 制定 caption
            if emo_type == "frown":
                caption = 'a person is frowning'
            elif emo_type == "gape":
                caption = 'a person is gaping'
            elif emo_type == "girl_smile":
                caption = 'a girl is smiling'
            elif emo_type == "pout":
                caption = 'a person is pouting'
            elif emo_type == "raise_eyebrows":
                caption = 'a persion is raising eyebrows'
            elif emo_type == "roll_eyes":
                caption = 'a person is rolling eyes'
            elif emo_type == "smile":
                caption = 'a person is smiling'
            elif emo_type == "wink":
                caption = 'a person is winking'
            else:
                raise NotImplementedError
                       
            # make reader
            try:                
                if self.load_raw_resolution:   # 没有走这一支
                    # video_reader = VideoReader(video_path, ctx=cpu(0))
                    video_reader = imageio.mimread(video_path, memtest=False)

                else:
                    json_path = video_path.replace('.gif','.json')
                    with open(json_path) as f:
                        info = json.load(f)  # 这个就是landmark的json文件
                    
                    # video_reader = VideoReader(video_path, ctx=cpu(
                    #     0), width=self.resolution[1], height=self.resolution[0])
                    reader = imageio.mimread(video_path, memtest=False)
                    width, height =  reader[0].shape[:2]
                    # reader = imageio.get_reader(gif_input_path)
                    # width, height =  reader.get_meta_data()['source_size']
                    new_width, new_height = self.resolution
                    
                    
                    ldmks = info['frames'][0]['faces'][-1]['landmarks']
                    x_list, y_list = [x for x,y in ldmks], [y for x,y in ldmks]
                    x_min,x_max = int(min(x_list)), int(max(x_list))
                    y_min,y_max = int(min(y_list)), int(max(y_list))
                    h_delta = min(int(height-y_max), y_min)
                    crop_y1 = y_min - h_delta
                    crop_y2 = y_max + h_delta
                    h = crop_y2 -crop_y1
                    w_delta = (h - (x_max-x_min)) / 2
                    crop_x1 = int(x_min - w_delta)
                    crop_x2 = int(x_max + w_delta)
                    

                    # print('x_min,x_max', x_min,x_max)
                    # print('y_min,y_max', y_min,y_max)
                    # print(h,w_delta)

                    # scale = min(new_width/width, new_height/height)
                    # crop_width = int(width * scale)
                    # crop_height = int(height * scale)
                    # left = (width - crop_width) // 2
                    # top = (height - crop_height) // 2
                    # right = left + crop_width
                    # bottom = top + crop_height
                    video_reader = []
                    ldmk_reader = []
                    for f_idx, frame in enumerate(reader):
                        # cropped_frame = frame[top:bottom, left:right]
                        
                        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                        resized_frame = cv2.resize(cropped_frame, (new_height, new_width))

                        # 存储临时结果
                        # cv2.imwrite("image_resized_{}.jpg".format(str(i)), resized_frame)
                        # print("save resize images")

                        # 存储当前帧 ldmk 的一个图像
                        f_ldmks = info['frames'][f_idx]['faces'][-1]['landmarks']
                        f_ldmks_resized = []
                        # ldmk_count = 0
                        for x,y in f_ldmks:
                            xc = x - crop_x1
                            yc = y - crop_y1
                            scale_x = new_width / (crop_x2-crop_x1)
                            scale_y = new_height / (crop_y2-crop_y1)
                            xr = int(xc * scale_x)
                            yr = int(yc * scale_y)
                            f_ldmks_resized.append((xr, yr))
                            # ldmk_count += 1
                        # print("ldmk_count:", ldmk_count)
                        # ldmk_image = Image.new('RGB', (self.resolution[0], self.resolution[1]), color=(0, 0, 0))
                        ldmk_image = np.zeros_like(resized_frame)
                        ldmk_image = drawLandmark_multiple_nobox(ldmk_image, f_ldmks_resized)
                        ldmk_image = np.array(ldmk_image)  
                        
                        # imageio.imsave("/data00/fsq/AnimateDiff-diffusers14/vis_results/ldmk_resize_{}.jpg".format(str(f_idx)), np.array(ldmk_image))
                        # ldmk_image2 = Image.open("/data00/fsq/AnimateDiff-diffusers14/vis_results/ldmk_resize_{}.jpg".format(str(f_idx)))
                        # if images_are_identical(ldmk_image, ldmk_image2):
                        #     print("两个图像内容完全相同")
                        # else:
                        #     print("两个图像内容不相同")
                        # print(np.max(np.array(ldmk_image)))
                        # print(np.max(np.array(ldmk_image2)))
                        # difference = np.array(ldmk_image) - np.array(ldmk_image2)
                        # print("Difference:", np.abs(difference).sum())
                        # ldmk_image.save("ldmk_image_1_{}.jpg".format(str(f_idx)))
                        # ldmk_image.save("ldmk_image_2_{}.jpg".format(str(f_idx)))

                        video_reader.append(resized_frame)
                        ldmk_reader.append(ldmk_image)
                    
                if len(video_reader) < self.video_length or len(reader[0].shape) < 3:
                    print(
                        f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            
            # NOTE
            # 到指定的目录下面去获取 face parsing 数据
            # ['background', 'face', 'rb', 'lb', 're', 'le', 'nose', 'ulip', 'imouth', 'llip', 'hair']
            face_parsing_file_path = os.path.join(self.face_parsing_path, rel_fp[:-4], "label.npy")
            record_path = os.path.join(self.face_parsing_path, rel_fp[:-4], "record.json")
            if os.path.exists(face_parsing_file_path):
                
                face_parsing_array = np.load(face_parsing_file_path)
                with open(record_path) as record_f:
                    face_parsing_record = json.load(record_f)

                if face_parsing_array.shape[0] == len(video_reader):
                    pass
                else:
                    # wrong_list.append(rel_fp)
                    print("face parsing file dimension error!")
                    index += 1
                    continue
            else:
                # missing_list.append(rel_fp)
                print("face parsing file do not exists!")
                index += 1
                continue
            

            # sample strided frames
            all_frames = list(range(0, len(video_reader), frame_stride))
            if len(all_frames) < self.video_length:  # recal a max fs
                frame_stride = len(video_reader) // self.video_length
                assert (frame_stride != 0)
                all_frames = list(range(0, len(video_reader), frame_stride))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            try:
                # frames = video_reader.get_batch(frame_indices)
                frames = video_reader[rand_idx:rand_idx+self.video_length]
                ldmks = ldmk_reader[rand_idx:rand_idx+self.video_length]
                
                # NOTE
                face_parsings = face_parsing_array[rand_idx:rand_idx+self.video_length] 
                for record_idx in range(rand_idx, rand_idx+self.video_length):
                    assert face_parsing_record[str(record_idx)] == "success" 


                assert (len(frames) == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
                assert (face_parsings.shape[0] == self.video_length), f'{len(face_parsings.shape[0])}, self.video_length={self.video_length}'  # NOTE

                frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float()[:,:3,:,:]  # [t,h,w,c] -> [t,c,h,w]
        
                break # 在这里进行了 break，其它情况都会找下一条数据
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        
        # print('frames', frames.shape)
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] ==
                    self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        # frames = frames / 255

        # fps_ori = video_reader.get_avg_fps()
        # fps_clip = fps_ori // frame_stride
        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        # data = {'video': frames, 'caption': caption, 'path': video_path,
        #         'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        
        landmarks = torch.tensor(np.array(ldmks)).unsqueeze(-1).permute(0, 3, 1, 2).float() 
        landmarks = landmarks / 255

        face_parsings = torch.tensor(np.array(face_parsings)).unsqueeze(-1).permute(0, 3, 1, 2).float() 
        face_parsings_mask = face_parsings != 10
        face_parsings = face_parsings * face_parsings_mask
        face_parsings = face_parsings / 10

        if self.is_image:
            frames = frames[0]
            landmarks = landmarks[0]
            face_parsings = face_parsings[0]
        sample = dict(pixel_values=frames, landmarks=landmarks, text=caption, rel_fp=rel_fp, face_parsings=face_parsings)
        return sample
    
    def __len__(self):
        return len(self.metadata)
    
class CelebvText(Dataset):
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 ):
        self.meta_path = meta_path
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
        self._load_metadata()
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

        
        

    def _load_metadata(self):
        with open(self.meta_path, 'r') as txt_file:
            lines = txt_file.readlines()
            metadata = [x.strip() for x in lines]
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        # metadata['caption'] = metadata['name']
        # del metadata['name']
        self.metadata = metadata
        # self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = str(sample)
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        # nstage=len_fps_schedule + 1
        assert (len(frame_strides) == len(schedule) + 1)
        global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(
                f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride

    def get_fs_based_on_probs(self, frame_strides, probs):
        assert (len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(
                    self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert (isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path, rel_fp = self._get_video_path(sample)
            # caption = sample['caption']+self.trigger_word
            
            # make reader
            try:
                emotion_desc_filepath = video_path.replace('videos/celebvtext_6','descripitions/emotion').replace('.mp4','.txt')
                with open(emotion_desc_filepath, 'r') as txt_file:
                    lines = txt_file.readlines()
                    lines = [x.strip() for x in lines]
                captions = lines[0].split(',')
                if 'begin' in captions[0] and len(captions[0].split(' ')) < 5:
                    captions = captions[1:]
                if 'end' in captions[-1] and len(captions[-1].split(' ')) < 5:
                    captions = captions[:-1]
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(
                        0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(
                        f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            # sample strided frames
            all_frames = list(range(0, len(video_reader), frame_stride))
            if len(all_frames) < self.video_length:  # recal a max fs
                frame_stride = len(video_reader) // self.video_length
                assert (frame_stride != 0)
                all_frames = list(range(0, len(video_reader), frame_stride))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]

            # select a caption: 这里没有帧级别的标注，所以对时间均分取对应的caption了
            caption_idx = int(rand_idx / len(all_frames) * len(captions))
            caption = captions[caption_idx]

            try:
                frames = video_reader.get_batch(frame_indices).asnumpy()

                # process landmarks
                # input [t,h,w,c]
                if self.is_image:
                    landmarks, info = get_landmarks(frames[0:1])
                    assert (len(landmarks) > 0)
                else:
                    landmarks, info = get_landmarks(frames)   
                    assert (frames.shape[0] == len(landmarks)), f'{frames.shape[0]}, landmarks={len(landmarks)}' 
                    assert (frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
           
                # crop
                width, height =  frames[0].shape[:2]
                # reader = imageio.get_reader(gif_input_path)
                # width, height =  reader.get_meta_data()['source_size']
                new_width, new_height = self.resolution
                ldmks = info['frames'][0]['faces'][-1]['landmarks']
                x_list, y_list = [x for x,y in ldmks], [y for x,y in ldmks]
                x_min,x_max = int(min(x_list)), int(max(x_list))
                y_min,y_max = int(min(y_list)), int(max(y_list))
                h_delta = min(int(height-y_max), y_min)
                crop_y1 = y_min - h_delta
                crop_y2 = y_max + h_delta
                h = crop_y2 -crop_y1
                w_delta = (h - (x_max-x_min)) / 2
                crop_x1 = int(x_min - w_delta)
                crop_x2 = int(x_max + w_delta)
                new_frames = []
                new_landmarks = []
                for frame,landmark in zip(frames,landmarks):                    
                    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                    resized_frame = cv2.resize(cropped_frame, (new_height, new_width))
                    new_frames.append(resized_frame)
                    cropped_landmarks = landmark[crop_y1:crop_y2, crop_x1:crop_x2]
                    resized_landmarks = cv2.resize(cropped_landmarks, (new_height, new_width))
                    
                    new_landmarks.append(resized_landmarks)
                
                # process face parsing

                frames = torch.tensor(np.array(new_frames)).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
                landmarks = torch.tensor(np.array(new_landmarks)).unsqueeze(-1).permute(0, 3, 1, 2).float() 

                break
            except:
                # print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

             
        
        # print('frames', frames.shape)
        # print('landmarks', landmarks.shape)
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] ==
                    self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        

        frames = (frames / 255 - 0.5) * 2
        landmarks = landmarks / 255

        # fps_ori = video_reader.get_avg_fps()
        # fps_clip = fps_ori // frame_stride
        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        # data = {'video': frames, 'caption': caption, 'path': video_path,
        #         'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        
        
        if self.is_image:
            frames = frames[0]
            landmarks = landmarks[0]
        # print(caption)
        sample = dict(pixel_values=frames, landmarks=landmarks, text=caption)
        return sample
    
    def __len__(self):
        return len(self.metadata)

   
class JsonDataset(Dataset):
    def __init__(self,
                 meta_path,
                 data_dir,
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
        self.meta_path = meta_path
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
        self._load_metadata()
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

    '''
    {
    "name": "Text Label",
    "data": [
        {
            "video_path": "/data00/Datasets/Videos/smile_video_1600/video1/4982409.mp4",
            "num_frames": 183,
            "data": [
                {
                    "frame_index": 91,
                    "prompt": "two little girls laying in the grass with their arms up in the air"
                }
            ]
        },
        {
            "video_path": "/data00/Datasets/Videos/smile_video_1600/video1/7741779.mp4",
            "num_frames": 276,
            "data": [
                {
                    "frame_index": 138,
                    "prompt": "an african american man and woman embrace in front of a tree"
                }
            ]
        },
    '''
    def _load_metadata(self):
        with open(self.meta_path) as f:
            info = json.load(f)
        metadata = info['data']
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        # metadata['caption'] = metadata['name']
        # del metadata['name']
        self.metadata = metadata
        # self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = str(sample)
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        # nstage=len_fps_schedule + 1
        assert (len(frame_strides) == len(schedule) + 1)
        global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(
                f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride

    def get_fs_based_on_probs(self, frame_strides, probs):
        assert (len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(
                    self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert (isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = sample['video_path']
            data = sample['data']
            frame_middle_index = data[0]['frame_index']
            caption = data[0]['prompt']

            # video_path, rel_fp = self._get_video_path(sample)
            # caption = sample['caption']+self.trigger_word
            
            # make reader
            try:
                # emotion_desc_filepath = video_path.replace('videos/celebvtext_6','descripitions/emotion').replace('.mp4','.txt')
                # with open(emotion_desc_filepath, 'r') as txt_file:
                #     lines = txt_file.readlines()
                #     lines = [x.strip() for x in lines]
                # captions = lines[0].split(',')
                # if 'begin' in captions[0] and len(captions[0].split(' ')) < 5:
                #     captions = captions[1:]
                # if 'end' in captions[-1] and len(captions[-1].split(' ')) < 5:
                #     captions = captions[:-1]
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(
                        0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(
                        f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            # sample strided frames
            all_frames = list(range(0, len(video_reader), frame_stride))
            if len(all_frames) < self.video_length:  # recal a max fs
                frame_stride = len(video_reader) // self.video_length
                assert (frame_stride != 0)
                all_frames = list(range(0, len(video_reader), frame_stride))

            # select a random clip
            # rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_start_index = frame_middle_index #- self.video_length // 2
            frame_start_index = random.randint(0,10)
            frame_indices = all_frames[frame_start_index:frame_start_index+self.video_length]

            # select a caption: 这里没有帧级别的标注，所以对时间均分取对应的caption了
            # caption_idx = int(rand_idx / len(all_frames) * len(captions))
            # caption = captions[caption_idx]

            try:

                json_path = video_path.replace('.mp4','_landmarks.json')
                # if not os.path.exists(json_path):
                #     print(f'landmark {json_path} not exist!!!!!')

                with open(json_path) as f:
                    info = json.load(f)  # 这个就是landmark的json文件

                frames = video_reader.get_batch(frame_indices).asnumpy()
                # process landmarks
                # input [t,h,w,c]
                # if self.is_image:
                #     landmarks, info = get_landmarks(frames[0:1])
                #     assert (len(landmarks) > 0)
                # else:
                #     landmarks, info = get_landmarks(frames)   
                #     assert (frames.shape[0] == len(landmarks)), f'{frames.shape[0]}, landmarks={len(landmarks)}' 
                #     assert (frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
           
                # crop
                width, height =  frames[0].shape[:2]
                # reader = imageio.get_reader(gif_input_path)
                # width, height =  reader.get_meta_data()['source_size']
                new_width, new_height = self.resolution
                # print('process landmarks')
                ldmks = info['frames'][frame_start_index]['faces'][0]['landmarks']
                x_list, y_list = [x for y,x in ldmks], [y for y,x in ldmks]
                x_min,x_max = int(min(x_list)), int(max(x_list))
                y_min,y_max = int(min(y_list)), int(max(y_list))
                h_delta = min(int(height-y_max), y_min, (y_max-y_min)//2)
                # h_delta = min(int(height-y_max), y_min)
                crop_y1 = max(y_min - h_delta, 0)
                crop_y2 = min(y_max + h_delta, height)
                h = crop_y2 -crop_y1
                w_delta = max((h - (x_max-x_min)) / 2, 0)
                crop_x1 = max(int(x_min - w_delta),0)
                crop_x2 = min(int(x_max + w_delta),width)
                
                new_frames = []
                new_landmarks = []
                # print('new_landmarks', len(new_landmarks), len(frames))
                for frame_idx, frame in enumerate(frames):
                    f_idx = frame_idx * frame_stride + frame_start_index
                    # cropped_frame = frame[top:bottom, left:right]
                    # print('frame', frame.shape, frame.max(), frame.min())
                    # print(crop_y1,crop_y2, crop_x1,crop_x2)
                    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                    # print('cropped_frame', cropped_frame.shape, cropped_frame.max())
                    resized_frame = cv2.resize(cropped_frame, (new_height, new_width))

                    # 存储临时结果
                    # cv2.imwrite("image_resized_{}.jpg".format(str(i)), resized_frame)
                    # print("save resize images")

                    # 存储当前帧 ldmk 的一个图像
                    f_ldmks = info['frames'][f_idx]['faces'][0]['landmarks']
                    # print('f_ldmks', f_ldmks)
                    f_ldmks_resized = []
                    # ldmk_count = 0
                    frame_ldmks = np.zeros((new_height, new_width,1))
                    # print('debug')
                    for x,y in f_ldmks:
                        xc = x - crop_x1
                        yc = y - crop_y1
                        scale_x = new_width / (crop_x2-crop_x1)
                        scale_y = new_height / (crop_y2-crop_y1)
                        xr = int(xc * scale_x)
                        yr = int(yc * scale_y)
                        # if yr >= new_height or yr < 0 or xr >= new_width or xr < 0:
                        #     continue
                        frame_ldmks[yr,xr] = 1
                        # print(yr,xr)
                    new_landmarks.append(frame_ldmks)  
                    new_frames.append(resized_frame)
                # print('new_landmarks', len(new_landmarks))

                

                # new_landmarks = []
                # new_frames = []
                # for frame_idx, frame in enumerate(frames):
                #     f_idx = frame_idx * frame_stride + frame_start_index
                #     width, height =  frame.shape[:2]
                #     ldmks = info['frames'][f_idx]['faces'][0]['landmarks']
                #     x_list, y_list = [x for x,y in ldmks], [y for x,y in ldmks]
                #     x_min,x_max = int(min(x_list)), int(max(x_list))
                #     y_min,y_max = int(min(y_list)), int(max(y_list))
                #     # h_delta = min(int(height-y_max), y_min)
                #     h_delta = min(int(height-y_max), y_min, (y_max-y_min)//2)
                #     crop_y1 = y_min - h_delta
                #     crop_y2 = y_max + h_delta
                #     h = crop_y2 -crop_y1
                #     w_delta = (h - (x_max-x_min)) / 2
                #     crop_x1 = int(x_min - w_delta)
                #     crop_x2 = int(x_max + w_delta)

                #     cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                #     resized_frame = cv2.resize(cropped_frame, (new_height, new_width))
                    
                #     frame_ldmks = np.zeros((new_height, new_width,1))
                #     print('debug')
                #     for x,y in ldmks:
                #         xc = x - crop_x1
                #         yc = y - crop_y1
                #         scale_x = new_width / (crop_x2-crop_x1)
                #         scale_y = new_height / (crop_y2-crop_y1)
                #         xr = int(xc * scale_x)
                #         yr = int(yc * scale_y)
                #         frame_ldmks[yr,xr] = 1
                #         print(yr,xr)
                #     new_landmarks.append(frame_ldmks)
                #     new_frames.append(resized_frame) 

                # process face parsing

                # print('new_frames', new_frames[0].shape)
                # print('landmarks1', new_landmarks[0].shape)
                frames = torch.tensor(np.array(new_frames)).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
                landmarks = torch.tensor(np.array(new_landmarks)).permute(0, 3, 1, 2).float() 

                break
            except:
                # print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

             
        
        # print('frames', frames.shape)
        # print('landmarks', landmarks.shape)
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] ==
                    self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        

        frames = (frames / 255 - 0.5) * 2
        landmarks = landmarks / 1

        # fps_ori = video_reader.get_avg_fps()
        # fps_clip = fps_ori // frame_stride
        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        # data = {'video': frames, 'caption': caption, 'path': video_path,
        #         'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        
        
        if self.is_image:
            frames = frames[0]
            landmarks = landmarks[0]
            print('frames', frames.shape)
            print('landmarks', landmarks.shape)
        
        sample = dict(pixel_values=frames, landmarks=landmarks, text=caption)
        print(f'debug caption:{caption}, pixel_values:{frames.shape}, landmarks:{landmarks.shape}')
        return sample
    
    def __len__(self):
        return len(self.metadata)
    
class EmotionsGif(Dataset):
    """
    EmotionsGif Dataset.
    Assumes EmotionsGif data is structured as follows.
    collected_emotions_gif/
        Frown/
            1.gif
            2.gif
        Gape/
            1.gif
            2.gif
        girl_smile_gif/
            asia_girl_smile_1.gif
            asia_girl_smile_2.gif
            ...
    """

    def __init__(self,
                 meta_path,
                 data_dir,
                 emotions_type,
                 subsample=None,
                 video_length=8,
                 resolution=[256, 256],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 ):
        self.meta_path = meta_path
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
        
        # self.all_emotions_type = ["all", "frown", "gape", "girl_smile_gif", "Pout", "Raise_eyebrows", "Roll_eyes", "smile", "wink"]
        self.all_emotions_type = os.listdir(self.data_dir)
        assert (emotions_type == "all" or emotions_type in self.all_emotions_type)
        self.emotions_type = emotions_type

        if meta_path != None:
            self._load_metadata()
        else:
            if emotions_type != "all":
                self.metadata = [emotions_type + '/' + gif_file for gif_file in os.listdir(os.path.join(self.data_dir, emotions_type))]
            else:
                self.metadata = []
                for emo_type in self.all_emotions_type:
                    self.metadata.extend([emo_type + '/' + gif_file for gif_file in os.listdir(os.path.join(self.data_dir, emo_type))])
        
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

    def _load_metadata(self):
        with open(self.meta_path, 'r') as txt_file:
            lines = txt_file.readlines()
            metadata = [x.strip() for x in lines]
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        # metadata['caption'] = metadata['name']
        # del metadata['name']
        self.metadata = metadata  # 就是一个列表
        # self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = str(sample) #+ '.gif'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_fs_based_on_schedule(self, frame_strides, schedule):
        # nstage=len_fps_schedule + 1
        assert (len(frame_strides) == len(schedule) + 1)
        global_step = self.counter // self.bs_per_gpu  # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(
                f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride

    def get_fs_based_on_probs(self, frame_strides, probs):
        assert (len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)

    def __getitem__(self, index):

        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(
                    self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(
                    self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert (isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            # video_path, rel_fp = self._get_video_path(sample)
            # caption = sample['caption']+self.trigger_word
            
            video_path = os.path.join(self.data_dir, sample)
            rel_fp = sample

            if self.emotions_type == "all":
                emo_type = video_path.split('/')[-2]
            else:
                emo_type = self.emotions_type

            # 制定 caption
            if emo_type == "frown":
                caption = 'a person is frowning'
            elif emo_type == "gape":
                caption = 'a person is gaping'
            elif emo_type == "girl_smile":
                caption = 'a girl is smiling'
            elif emo_type == "pout":
                caption = 'a person is pouting'
            elif emo_type == "raise_eyebrows":
                caption = 'a persion is raising eyebrows'
            elif emo_type == "roll_eyes":
                caption = 'a person is rolling eyes'
            elif emo_type == "smile":
                caption = 'a person is smiling'
            elif emo_type == "wink":
                caption = 'a person is winking'
            else:
                raise NotImplementedError
                       
            # make reader
            try:                
                if self.load_raw_resolution:   # 没有走这一支
                    # video_reader = VideoReader(video_path, ctx=cpu(0))
                    video_reader = imageio.mimread(video_path)

                else:
                    json_path = video_path.replace('.gif','.json')
                    with open(json_path) as f:
                        info = json.load(f)  # 这个就是landmark的json文件
                    
                    # video_reader = VideoReader(video_path, ctx=cpu(
                    #     0), width=self.resolution[1], height=self.resolution[0])
                    reader = imageio.mimread(video_path)
                    width, height =  reader[0].shape[:2]
                    # reader = imageio.get_reader(gif_input_path)
                    # width, height =  reader.get_meta_data()['source_size']
                    new_width, new_height = self.resolution
                    
                    
                    ldmks = info['frames'][0]['faces'][-1]['landmarks']
                    x_list, y_list = [x for x,y in ldmks], [y for x,y in ldmks]
                    x_min,x_max = int(min(x_list)), int(max(x_list))
                    y_min,y_max = int(min(y_list)), int(max(y_list))
                    h_delta = min(int(height-y_max), y_min)
                    crop_y1 = y_min - h_delta
                    crop_y2 = y_max + h_delta
                    h = crop_y2 -crop_y1
                    w_delta = (h - (x_max-x_min)) / 2
                    crop_x1 = int(x_min - w_delta)
                    crop_x2 = int(x_max + w_delta)
                    

                    # print('x_min,x_max', x_min,x_max)
                    # print('y_min,y_max', y_min,y_max)
                    # print(h,w_delta)

                    # scale = min(new_width/width, new_height/height)
                    # crop_width = int(width * scale)
                    # crop_height = int(height * scale)
                    # left = (width - crop_width) // 2
                    # top = (height - crop_height) // 2
                    # right = left + crop_width
                    # bottom = top + crop_height
                    video_reader = []
                    for frame in reader:
                        # cropped_frame = frame[top:bottom, left:right]
                        
                        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                        resized_frame = cv2.resize(cropped_frame, (new_height, new_width))
                        video_reader.append(resized_frame)
                if len(video_reader) < self.video_length or len(reader[0].shape) < 3:
                    print(
                        f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            # sample strided frames
            all_frames = list(range(0, len(video_reader), frame_stride))
            if len(all_frames) < self.video_length:  # recal a max fs
                frame_stride = len(video_reader) // self.video_length
                assert (frame_stride != 0)
                all_frames = list(range(0, len(video_reader), frame_stride))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                # frames = video_reader.get_batch(frame_indices)
                frames = video_reader[rand_idx:rand_idx+self.video_length]

                assert (
                    len(frames) == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
                frames = torch.tensor(np.array(frames)).permute(
                    0, 3, 1, 2).float()[:,:3,:,:]  # [t,h,w,c] -> [t,c,h,w]
        
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        
        # print('frames', frames.shape)
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2] == self.resolution[0] and frames.shape[3] ==
                    self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        # frames = frames / 255

        # fps_ori = video_reader.get_avg_fps()
        # fps_clip = fps_ori // frame_stride
        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        # data = {'video': frames, 'caption': caption, 'path': video_path,
        #         'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        
        
        if self.is_image:
            frames = frames[0]
        sample = dict(pixel_values=frames, text=caption, rel_fp=rel_fp)
        return sample
    
    def __len__(self):
        return len(self.metadata)



if __name__ == '__main__':
    # local_path = '/data/users/liangjiajun/Datasets/'
    # mount_path = '/dataset00/'
    # meta_path = mount_path + '/Videos/CelebV-Text/descripitions/remains_happy_filelist.txt'
    # data_dir = mount_path + '/Videos/CelebV-Text/videos/celebvtext_6'
    # frame_stride = 1
    # is_image = False
    # num_workers = 2
    # dataset = EmotionsGif_controlnet(
    #     meta_path=None,
    #     data_dir="/dataset00/Videos/collected_emotions_gif_codeformer",
    #     emotions_type="smile",
    #     is_image=False,
    #     face_parsing_path="/dataset00/Videos/collected_emotions_gif_face_parsing/")
    # print('train_dataset size is:', len(dataset))  # 1044

    # # dataset = CelebvText(meta_path, data_dir, frame_stride=frame_stride, is_image=is_image)
    # # print('dataset size is ', len(dataset))

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, num_workers=16,)
    
    # for idx, batch in enumerate(dataloader):
    #     caption = batch["text"][0]
        
    #     print(batch["text"])
    #     print(batch["pixel_values"].shape, batch["pixel_values"].max(), batch["pixel_values"].min())
    #     print(batch["landmarks"].shape, batch["landmarks"].max(), batch["landmarks"].min())
    #     print(batch["face_parsings"].shape, batch["face_parsings"].max(), batch["face_parsings"].min())
    #     if is_image:
    #         image_array = ((batch["pixel_values"][0].permute(1, 2, 0)+1)/2 * 255).numpy().astype(np.uint8)
    #         ldmks_array = ((batch["landmarks"][0].permute(1, 2, 0)) * 255).numpy().astype(np.uint8)
    #         cv2.imwrite(f"debug/{caption}_image.png", image_array)
    #         cv2.imwrite(f"debug/{caption}_ldmks.png", ldmks_array)

    #     else:
    #         video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
    #         ldmks_array = ((batch["landmarks"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
    #         face_parsings_array = ((batch["face_parsings"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)

    #         print('caption:', caption)
    #         with imageio.get_writer(f"debug/{idx}_{caption}_video.mp4", fps=30) as video_writer:
    #             for frame,ldmk,f_p in zip(video_array,ldmks_array,face_parsings_array):
    #                 ldmk = np.repeat(ldmk, repeats=3, axis=2)
    #                 f_p = np.repeat(f_p, repeats=3, axis=2)
    #                 res = np.hstack((frame,ldmk,f_p))
    #                 video_writer.append_data(res)
            
    #         # with imageio.get_writer(f"debug/{caption}_ldmks.mp4", fps=30) as video_writer:
    #         #     for frame in ldmks_array:
    #         #         video_writer.append_data(frame)
            
    #         # with imageio.get_writer(f"debug/{caption}_face_parsings.mp4", fps=30) as video_writer:
    #         #     for frame in face_parsings_array:
    #         #         video_writer.append_data(frame)

    #     if idx >= 20:
    #         break

    local_path = '/data/users/liangjiajun/Datasets/'
    mount_path = '/dataset00'
    meta_path = mount_path + '/Videos/smile_video_1600/video1/text_label.json'
    data_dir = mount_path + '/Videos/CelebV-Text/videos/celebvtext_6'
    frame_stride = 1
    is_image = False
    num_workers = 2
    dataset = JsonDataset(meta_path, data_dir, frame_stride=frame_stride, is_image=is_image)
    # print('dataset size is ', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1,)
    
    for idx, batch in enumerate(dataloader):
        caption = batch["text"][0]
        
        print(batch["pixel_values"].shape, batch["text"])
        print(batch["pixel_values"].max(), batch["pixel_values"].min())
        if is_image:
            image_array = ((batch["pixel_values"][0].permute(1, 2, 0)+1)/2 * 255).numpy().astype(np.uint8)
            ldmks_array = ((batch["landmarks"][0].permute(1, 2, 0)) * 255).numpy().astype(np.uint8)
            cv2.imwrite(f"debug/{caption}_image.png", image_array)
            cv2.imwrite(f"debug/{caption}_ldmks.png", ldmks_array)

        else:
            batch["pixel_values"] = batch["pixel_values"] * (1-batch["landmarks"])
            video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
            ldmks_array = ((batch["landmarks"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)

            
            print('caption:', caption)
            with imageio.get_writer(f"debug/{idx}_{caption}_video.mp4", fps=30) as video_writer:
                for frame in video_array:
                    video_writer.append_data(frame)
            
            with imageio.get_writer(f"debug/{idx}_{caption}_ldmks.mp4", fps=30) as video_writer:
                for frame in ldmks_array:
                    video_writer.append_data(frame)

        if idx >= 20:
            break
