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
            emotion_desc_filepath = video_path.replace('videos/celebvtext_6','descripitions/emotion').replace('.mp4','.txt')
            with open(emotion_desc_filepath, 'r') as txt_file:
                lines = txt_file.readlines()
                lines = [x.strip() for x in lines]
            captions = lines[0].split(',')
            if 'begin' in captions[0] and len(captions[0].split(' ')) < 5:
                captions = captions[1:]
            if 'end' in captions[-1] and len(captions[-1].split(' ')) < 5:
                captions = captions[:-1]
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

            # select a caption: 这里没有帧级别的标注，所以多时间均分取对应的caption了
            caption_idx = int(rand_idx / len(all_frames) * len(captions))
            caption = captions[caption_idx]

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


if __name__ == '__main__':
    local_path = '/data/users/liangjiajun/Datasets/'
    mount_path = '/dataset00/'
    meta_path = mount_path + '/Videos/CelebV-Text/descripitions/remains_happy_filelist.txt'
    data_dir = mount_path + '/Videos/CelebV-Text/videos/celebvtext_6'
    frame_stride = 4
    dataset = CelebvText(meta_path, data_dir, frame_stride=frame_stride)
    # print('dataset size is ', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["text"])
        
        # print(batch["pixel_values"].shape, len(batch["text"]))
        # print(batch["pixel_values"].max(), batch["pixel_values"].min())
        # video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
        # caption = batch["text"][0]
        # with imageio.get_writer(f"{caption}.mp4", fps=30) as video_writer:
        #     for frame in video_array:
        #         video_writer.append_data(frame)

        if idx >= 30:
            break
