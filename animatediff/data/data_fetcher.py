import orjson
import msgpack
from megfile import s3_save_as
# import webp
# import lmdb
import zipfile
# from balls import p_runner
from torch.utils.data import Dataset, dataloader
from tqdm import tqdm
import tarfile
import megfile
from megfile import smart_open
import os
import refile
import numpy as np
import cv2
from PIL import Image
import json
import numpy as np
from collections import defaultdict
# import ripplit as rp
import re
import nltk
from contextlib import contextmanager
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from line_profiler import LineProfiler
from megfile import smart_remove

# 下载NLTK停用词
import logging

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Writes to console
logger.setLevel(logging.CRITICAL)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)


# 将字典数据 dump 为 msgpack 格式


def load_msgpack_list(file_path: str):
    loaded_data = []
    with smart_open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data


def dump_orjson(obj, json_path: str, show_info=True):
    obj_data = orjson.dumps(obj)
    with megfile.smart_open(json_path, 'wb') as w_file:
        w_file.write(obj_data)


def load_orjson(json_path: str):
    with megfile.smart_open(json_path, 'rb') as r_file:
        res = orjson.loads(r_file.read())
    return res


def load_tar(p):

    return tarfile.open(fileobj=megfile.smart_open(p, 'rb'))


class AutoSplitMsgPacker():

    def __init__(self, out_dir, split_size=1e4,verbose = False,overwrite = False):
        
        self.verbose = verbose
        self.packer = msgpack.Packer()
        self.split_file_count = 0
        self.obj_count = 0
        self.split_size = split_size
        self.overwrite = overwrite

        assert not megfile.smart_exists(out_dir) or (
            megfile.smart_exists(out_dir) and megfile.smart_isdir(out_dir))
        self.out_dir = out_dir
        self.init_io()

    def init_io(self):

        self.outf = os.path.join(
            self.out_dir, f'{self.split_file_count}.msgpack')
        
        # its expensive to check file exists on oss
        self.fio = smart_open(self.outf, 'wb')

    def close(self):
        if self.fio is not None:
            self.fio.close()
        
    def increase_count(self):
        self.obj_count +=1

    def write(self, obj: dict):

        if self.obj_count >= self.split_size:
            self.fio.close()
            if self.verbose:
                print(f"Write {self.obj_count} objs into {self.outf}, done...")
            self.split_file_count += 1
            self.obj_count = 0
            self.init_io()

        self.fio.write(self.packer.pack(obj))
        self.obj_count += 1


def uncompress_oss_file(src_dir, trg_dir, filter_type='jpg'):

    if "*" in src_dir:

        compresssed_files = megfile.smart_glob(
            src_dir
        )

        for f in tqdm(compresssed_files):

            fpath, fname = os.path.split(f)
            trg_subdir = os.path.join(trg_dir, fname)
            tar_obj = load_tar(f)
            for mem in tqdm(tar_obj.getmembers()):

                if filter_type is not None and not mem.name.lower().endswith(filter_type):
                    continue

                tar_uncompress_file = os.path.join(trg_subdir, mem.name)
                with smart_open(tar_uncompress_file, 'wb') as wf:
                    wf.write(tar_obj.extractfile(mem.name).read())

from functools import lru_cache
@lru_cache(maxsize = 4)
def load_tar(p): return tarfile.open(fileobj=smart_open(p, 'rb'))


@contextmanager
def load_split_tar_gz(files, tmp_file=None, delete_tmp=False):

    if not tmp_file:
        last_dir = os.path.split(files[0])[0]
        tmp_file = os.path.join(last_dir, 'tmp.tar.gz')

    if megfile.smart_exists(tmp_file):
        print(f'Merging tar files has been done : {tmp_file}')
    else:
        print(f'Merge tar files : {tmp_file}')
        with smart_open(tmp_file, 'ab') as target:
            # 遍历源文件列表
            for source_filename in tqdm(files):
                # 打开每个源文件以二进制读取模式
                with smart_open(source_filename, 'rb') as source:
                    # 读取源文件的内容，并追加到目标文件中
                    target.write(source.read())
    yield
    if delete_tmp:
        smart_remove(tmp_file)


def filter_nonsense_words(text):
    # 转换为小写字母
    text = text.lower()

    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text)

    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 去除非字母字符
    text = re.sub(r'[^a-z ]+', ' ', text)

    # 分词
    words = word_tokenize(text)
    def is_noun(pos): return pos[:2] == 'NN'

    # 只保留名词
    words = [word for (word, pos) in nltk.pos_tag(words) if is_noun(pos)]

    # # 移除停用词
    # stop_words = set(stopwords.words("english"))
    # words = [word for word in words if word not in stop_words]

    # 返回处理后的文本
    return ' '.join(words)


def preprocess_caption(caption, split_each_word=False):
    caption = filter_nonsense_words(caption)
    caption = caption.strip()
    if split_each_word:
        caption = caption.replace(" ", ". ")
    return caption


def load_json(f):

    with smart_open(f, 'r') as rf:
        return json.loads(rf.read())


def load_img(f):
    with smart_open(f, 'rb') as rf:
        return cv2.imdecode(np.frombuffer(rf.read(), dtype='uint8'), 1)


def load_img_from_mount_tar(file, mount_dir = '/data/users/weisiyuan/laion2B-en-aesthetic-data'):

    # volces-tos:weisiyuan/dataset/laion2B-en-aesthetic-data
    
    tar_fname = file.split("/")[-2]
    img_fname = file.split("/")[-1]
    tar_fname = os.path.join(mount_dir,tar_fname)
    tar_obj = load_tar(tar_fname)
    img = Image.open(tar_obj.extractfile(img_fname)).convert("RGB")
    return np.array(img),tar_fname,img_fname

def dump_img(arr, f):
    with smart_open(f, 'wb') as wf:
        postfix = f.split(".")[-1].lower()
        content = cv2.imencode("."+postfix, arr)[1].tobytes()
        wf.write(content)

# # aws configure
# # aws configure set s3.endpoint_url http://oss.i.basemind.com


COLORS_WORDS = ['red', 'blue', 'green', 'yellow', 'purple','dark',
                'orange', 'pink', 'brown', 'black', 'white', 'gray']
BLACK_WORDS = [
    'group', 'front', 'back', 'couple', 'top', 'left', 'right','set','herd',

    'field',
    "beauty",
    "brightness",
    "charm",
    "color",
    "elegance",
    "grace",
    "harmony",
    "light",
    "magic",
    "warmth"
] + COLORS_WORDS

BLACK_WORDS = set(BLACK_WORDS)


def split_nouns(ori_caption, min_caption_len=10):
    text = ori_caption.lower()
    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除非字母字符
    text = re.sub(r'[^a-z ]+', ' ', text)
    # 分词
    words = word_tokenize(text)

    if len(words) < min_caption_len:
        return None

    def is_noun(pos): return pos[:2] == 'NN'
    noun_words = [word for (word, pos) in nltk.pos_tag(
        words) if is_noun(pos) and word.lower() not in BLACK_WORDS]
    noun_words = " . ".join(noun_words) + " ."
    return noun_words


def preprocess(inps):
    img_path, meta_path = inps
    with megfile.smart_open(img_path, 'rb') as fio:
        img = Image.open(fio).convert("RGB")
    with megfile.smart_open(meta_path, 'r') as fio:
        meta = json.load(fio)

    return img, meta, meta['caption'], img_path


def load_ffhq(min_size=512, min_caption_len=1, batch_size=32, parallel=128):

    root_dir = "s3://narutowei/datasets/FFHQ/raw/"

    src_files = megfile.smart_glob(os.path.join(root_dir, '*.tar.gz.*'))

    with load_split_tar_gz(src_files, delete_tmp=False):
        import ipdb
        ipdb.set_trace()
        yield None


def uncompress_files(source_path, target_path):
    fp_file = smart_open(source_path, 'rb')

    zip_file = zipfile.ZipFile(fp_file)

    for zipinfo, zipname in zip(zip_file.infolist(), zip_file.namelist()):
        with zip_file.open(zipinfo) as source:
            if not zipinfo.filename.endswith('/'):
                s3_save_as(source, os.path.join(target_path, zipname))

    zip_file.close()


def uncompress_mdb(source_path, flat=False, limit=-1):

    data_mdb_files = refile.smart_glob(os.path.join(
        source_path, '**', 'data.mdb'), recursive=True)

    for mdb_f in tqdm(data_mdb_files):
        with smart_open(mdb_f, 'rb') as rf:
            env = lmdb.open(rf, map_size=1099511627776,
                            max_readers=100, readonly=True)

        out_dir = os.path.split(mdb_f)[0]
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, val in cursor:

                image_out_path = os.path.join(
                    image_out_dir, key.decode('ascii') + '.webp')
                webp_data = webp.WebPData.from_buffer(f.read())
                arr = webp_data.decode(color_mode=WebPColorMode.BGR)
                with smart_open(image_out_path, 'wb') as fp:
                    content = cv2.imencode('.png', arr)[1].tobytes()
                    fp.write(content)

from glob import glob       
def load_laion2b_from_msgpack_infos(file_path):
    
    meta_files = glob(
            os.path.join(file_path, "*.msgpack"))
    meta_info_list = []
    for meta_f in meta_files:
        meta_info_list =  load_msgpack_list(meta_f)
        for info in meta_info_list:

            if info.get('labels',None) and info.get('boxes',None):
                yield info
    
def batch_collabrator( gen , bs):
    
    batch = None
    for info in gen:
        
        if isinstance(info,tuple):
            
            if batch is None:
                batch = [[] for _ in range(len(info))]
            for i,item in enumerate(info):
                batch[i].append(item)
            
            if len(batch[0]) == bs:
                yield batch
                batch = [[] for _ in range(len(info))]
        else:
            if batch is None:
                batch = []
            
            batch.append(info)
            if len(batch) ==  bs:
                yield batch
                batch = []
    


def laion2b_aesthetic_batch_gen(min_size=512, min_caption_len=1, batch_size=32, parallel=128):

    # laion2b_aesthetic_data = "s3://ljj/laion2B-en-aesthetic-data/"
    # laion2b_aesthetic_data = 's3://narutowei/laion2B-en-aesthetic-data/'
    laion2b_aesthetic_data = 's3://weisiyuan-dev/laion2B-en-aesthetic-data/'
    batch_img = []
    batch_caption = []
    batch_nouns = []
    batch_img_paths = []
    for tar_file in megfile.smart_glob(os.path.join(laion2b_aesthetic_data, '*.tar')):

        is_tar = False
        if megfile.smart_isfile(tar_file):
            is_tar = True
            tar_obj = load_tar(tar_file)
            file_list = [mem.name for mem in tar_obj.getmembers()]
        else:
            file_list = list(megfile.smart_glob(os.path.join(tar_file, '*')))

        img_list = sorted(
            list(filter(lambda p: p.lower().endswith(".jpg"), file_list)))
        json_list = sorted(
            list(filter(lambda p: p.lower().endswith(".json"), file_list)))
        n_total = len(file_list)

        for img, meta, ori_caption, img_path in p_runner(preprocess, zip(img_list, json_list), runner=parallel, mode='builtin_ThreadPool', process_bar=False):
            if img.height < min_size and img.width < min_size:
                continue
            if meta['punsafe'] > 0.1:
                continue

            batch_img.append(img)
            batch_caption.append(ori_caption)
            batch_img_paths.append(img_path)

            if all([
                    len(batch) == batch_size for batch in [batch_img, batch_caption, batch_img_paths]]):
                yield batch_img, batch_caption, batch_img_paths
                batch_img, batch_caption, batch_img_paths = [], [], []

        # ori_caption = meta['caption']
        # text = ori_caption.lower()
        # # 去除URL
        # text = re.sub(r'http\S+|www.\S+', '', text)
        # # 去除HTML标签
        # text = re.sub(r'<.*?>', '', text)
        # # 去除非字母字符
        # text = re.sub(r'[^a-z ]+', ' ', text)
        # # 分词
        # words = word_tokenize(text)

        # if len(words) < min_caption_len:
        #     return None

        # def is_noun(pos): return pos[:2] == 'NN'
        # noun_words = [word for (word, pos) in nltk.pos_tag(
        #     words) if is_noun(pos)]
        # noun_words = ".".join(noun_words)

        # for i, (img_path, meta_path) in enumerate(zip(img_list, json_list)):
        #     if is_tar:
        #         img = Image.open(tar_obj.extractfile(img_path)).convert("RGB")
        #         meta = json.load(tar_obj.extractfile(meta_path))
        #     else:
        #         with megfile.smart_open(img_path, 'rb') as fio:
        #             img = Image.open(fio).convert("RGB")
        #         with megfile.smart_open(meta_path, 'r') as fio:
        #             meta = json.load(fio)
        #     h, w = img.height, img.width
        #     if h < min_size and w < min_size:
        #         continue

        #     if meta['punsafe'] > 0.1:
        #         continue

        #     ori_caption = meta['caption']
        #     text = ori_caption.lower()
        #     # 去除URL
        #     text = re.sub(r'http\S+|www.\S+', '', text)
        #     # 去除HTML标签
        #     text = re.sub(r'<.*?>', '', text)
        #     # 去除非字母字符
        #     text = re.sub(r'[^a-z ]+', ' ', text)
        #     # 分词
        #     words = word_tokenize(text)

        #     if len(words) < min_caption_len:
        #         continue

        #     def is_noun(pos): return pos[:2] == 'NN'
        #     noun_words = [word for (word, pos) in nltk.pos_tag(
        #         words) if is_noun(pos)]
        #     noun_words = ".".join(noun_words)

        #     batch_nouns.append(noun_words)
        #     batch_img.append(img)
        #     batch_caption.append(ori_caption)
        #     batch_img_paths.append(f"{img_path}")

        #     if all([
        #             len(batch) == batch_size for batch in [batch_nouns, batch_img, batch_caption, batch_img_paths]]):
        #         yield batch_img, batch_nouns, batch_caption, batch_img_paths
        #         batch_img, batch_nouns, batch_caption = [], [], []
        #         batch_img_paths = []


# class Laion2BDataset(Dataset)

if __name__ == "__main__":

    count = 0

    for f in tqdm(refile.smart_glob('s3://narutowei/datasets/lsun/raw/objects/*.zip')):
        if count >= 7:
            try:
                uncompress_files(
                    f, "s3://narutowei/datasets/lsun/raw/objects/")
            except:
                pass
        count += 1
    # uncompress_mdb("s3://narutowei/datasets/lsun/raw/objects/")
    # for batch_img, batch_caption, batch_img_paths in tqdm(laion2b_aesthetic_batch_gen()):
    #     for item in batch_img, batch_caption, batch_img_paths:
    #         print(len(item))
# from wym的自采数据

# root_dir = 's3://wym/datasets/photos/'

# image_dir = os.path.join(root_dir,'images')
# meta_dir = os.path.join(root_dir,'meta')

# for img_subdir in megfile.smart_glob(os.path.join(image_dir,'photos.parquet_*')):

#     dirname = img_subdir.split("/")[-1]
#     meta_subdir = os.path.join(meta_dir,dirname)

#     if not megfile.smart_exists(meta_subdir):
#         continue

#     for meta_file in megfile.smart_glob(os.path.join(meta_subdir,'*.json')):

#         fname = meta_file.split("/")[-1].split(".")[0]
#         img_file = os.path.join(img_subdir,fname)

#         meta = load_json(meta_file)
#         img = load_img(img_file)

#         import ipdb;ipdb.set_trace()
