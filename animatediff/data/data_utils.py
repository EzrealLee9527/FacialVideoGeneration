import os
import megfile
from megfile import smart_open, smart_exists
import json
import tarfile
from copy import deepcopy
import pandas as pd
from decord import VideoReader, cpu
import imageio
import time
from megfile.smart_path import SmartPath
import msgpack
import hashlib
import pickle

def local_file_cache_func(local_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 创建本地路径
            if not os.path.exists(local_path):
                os.makedirs(local_path)

            # 为函数输入创建哈希值
            hasher = hashlib.sha256()
            hasher.update(repr((func.__name__, args, kwargs)).encode())
            file_name = f"{hasher.hexdigest()}.cache"

            cache_file_path = os.path.join(local_path, file_name)

            # 如果缓存文件存在，则加载并返回结果
            if os.path.exists(cache_file_path):
                with smart_open(cache_file_path, "rb") as cache_file:
                    result = pickle.load(cache_file)
                print("Loaded result from cache")
            else:
                # 否则，计算结果并将其保存到缓存文件中
                result = func(*args, **kwargs)
                with smart_open(cache_file_path, "wb") as cache_file:
                    pickle.dump(result, cache_file)
                print("Saved result to cache")

            return result

        return wrapper
    return decorator

def check_file_exists(s3_path):
    path = SmartPath(s3_path)
    return path.exists()

# load metafile
def load_csv(csv_path):
    with smart_open(csv_path, 'rb') as r_file:
        res = pd.read_csv(r_file)
    return res

def load_json(json_path: str):
    with smart_open(json_path, 'rb') as r_file:
        res = json.loads(r_file.read())
    return res

def load_txt(txt_path: str):
    with smart_open(txt_path, 'r') as txt_file:
        res = txt_file.readlines()
    return res

def get_video_reader(video_path):
    with smart_open(video_path, 'rb') as r_file:
        video_reader = VideoReader(r_file, ctx=cpu(0))
    return video_reader

def process_datas_to_tar():
    return

def get_file_from_tar():
    return

def vepfs_to_tos():
    return

def tos_to_vepfs():

    return

def shuffle():
    return

# process videos to landmarks
# 根据人脸处理数据
def get_landmarks_from_videos():

    return

def safe_crop_face():
    return

def get_captions():
    return


# TODO: 把多个label整合（caption、landmark、faceparsing），msgpack下
# 直接写一个多进程打label的吧
def multi_generate_labels():
    return

def load_msgpack_list(file_path: str):
    loaded_data = []
    with smart_open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f,strict_map_key = False)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data
    
if __name__ == '__main__':
    print('start test functions')
    # csv_path = 's3://ljj-sh/Datasets/Videos/webvid/results_2M_val.csv'
    # csv_data = load_csv(csv_path)
    # print(csv_data[:2])

    # time_start = time.time()
    # video_path = 's3://ljj-sh/Datasets/Videos_230905/CelebV-Text/videos/celebvtext_6/88e6FikYYGc_8_0.mp4'
    # # local_video_path = './88e6FikYYGc_8_0.mp4'
    # video_reader = get_video_reader(video_path)
    # frame_indices = list(range(0, len(video_reader), 1))
    # frames = video_reader.get_batch(frame_indices).asnumpy()
    # time_end = time.time()
    # print(f'read one video use {time_end-time_start}s')
    # with imageio.get_writer(f"test_video_reader.mp4", fps=30) as video_writer:
    #     for frame in frames:
    #         video_writer.append_data(frame)

    # video_path = 's3://ljj-sh/Datasets/Videos_230905/CelebV-Text/videos/celebvtext_6/no_exist.mp4'
    # print(check_file_exists(video_path))

    data_path = 's3://ljj-sh/Datasets/Videos/msgpacks/videos_231002/videos1600_gen-worker0-112.msgpack'
    msg_datas = load_msgpack_list(data_path)
    print(msg_datas)
    


