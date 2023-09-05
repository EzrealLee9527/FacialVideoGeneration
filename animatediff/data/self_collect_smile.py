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
        self.metadata =  metadata
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
        print(index)
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
            # index = index % len(self.metadata)
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
    '''
    invilid datas: 
        smile_149.gif
    '''

    local_path = '/data/users/liangjiajun/Datasets/'
    mount_path = '/dataset00/'
    meta_path = mount_path + '/Videos/smile/filelist.txt'
    data_dir = mount_path + '/Videos/smile/gifs'
    frame_stride = 1
    dataset = SmileGif(meta_path, data_dir, frame_stride=frame_stride)
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, num_workers=16,shuffle=False)
    valid_count = 0
    for idx, batch in enumerate(dataloader):
        valid_count += 1
        print(valid_count)
        # print(batch["pixel_values"].shape, batch["text"])
        # print(batch["pixel_values"].max(), batch["pixel_values"].min())
        # rel_fp= batch['rel_fp']
        # video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
        # caption = batch["text"][0]
        # with imageio.get_writer(f"{rel_fp[0]}_{caption}.mp4", fps=30) as video_writer:
        #     for frame in video_array:
        #         video_writer.append_data(frame)

        # if idx >= 20:
        #     break
