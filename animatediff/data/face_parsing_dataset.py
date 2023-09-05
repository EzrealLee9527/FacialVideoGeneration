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
from PIL import Image, ImageDraw
import pdb
from tqdm import tqdm

wrong_list = []
missing_list = []

def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

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
                 meta_path,
                 data_dir,
                 emotions_type,
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
            # index = index % len(self.metadata)
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
                        ldmk_image = Image.new('RGB', (self.resolution[0], self.resolution[1]), color=(0, 0, 0))
                        draw_landmarks(ldmk_image, f_ldmks_resized)
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
        
        ldmks = np.stack(ldmks, axis=0)  # numpy array, (L, 512, 512, 3)

        if self.is_image:
            frames = frames[0]
        sample = dict(pixel_values=frames, ldmks=ldmks, text=caption, rel_fp=rel_fp, face_parsings=face_parsings)
        return sample
    
    def __len__(self):
        return len(self.metadata)


if __name__ == "__main__":
    # 创建 datasets
    dataset = EmotionsGif_controlnet(
        meta_path=None,
        data_dir="/data00/Datasets/Videos/collected_emotions_gif_codeformer",
        emotions_type="girl_smile",
        is_image=False,
        face_parsing_path="/data00/Datasets/Videos/collected_emotions_gif_face_parsing/")
    print('train_dataset size is:', len(dataset))  # 1044

    # a = dataset[0]

    # pdb.set_trace()

    for i,data in tqdm(enumerate(dataset)):
        a = data

        pdb.set_trace()

    print("over!")
    print("wrong_list", wrong_list)
    print("missing_list", missing_list)