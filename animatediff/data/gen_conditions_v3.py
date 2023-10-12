
from data_fetcher import dump_img, load_img
from collections import defaultdict
from data_fetcher import AutoSplitMsgPacker, split_nouns
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
import pickle
from functools import reduce
import torchvision.transforms.functional as F_t
import torch.nn.functional as F
import megfile
from lavis.models import load_model_and_preprocess
import functools
import tarfile
from functools import lru_cache
import megfile
from datetime import datetime
# from balls.imgproc import imdecode
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import orjson
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import warnings
import torch.multiprocessing as mp
from copy import deepcopy
import cv2
from data_utils import *
import msgpack_numpy as m
from face_model import FaceAnalysis

m.patch()

DEBUG = os.environ.get('DEBUG')
if DEBUG:
    print('start debug mode')

def load_landmarks_model():
    face_ana = FaceAnalysis()
    face_ana.prepare(ctx_id=0, det_size=(640, 640))
    return face_ana

def gen_landmarks(img, face_ana):
    res = face_ana.get(img, 1)
    if DEBUG:
        img = face_ana.draw_on(img, res)
    # res[0] : dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
    return [dict(x) for x in res], img

def load_faceparsing_model():

    return

def gen_faceparsings():

    return

def load_blip_caption(device):
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    return model, vis_processors

# def load_blip_vqa(device):
#     model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
#     return  model, vis_processors, txt_processors

def gen_blip_vqa_captions(vis_processors, txt_processors, model, question_batch, image_batch):
    # question = "Which city is this photo taken?"
    # # use "eval" processors for inference
    # image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # question = txt_processors["eval"](question)
    # model.predict_answers(samples=samples, inference_method="generate")
    # samples = {"image": image, "text_input": question}

    # create a batch of questions, make sure the number of questions matches the number of images

    res = model.predict_answers(samples={"image": image_batch, "text_input": question_batch}, inference_method="generate")
    return res


def fetch_single_video(remotef,tmpf, trg_shape=(256, 256)):
    try:
        # os.system(f'rm {os.path.abspath(tmpf)}')
        video_reader = get_video_reader(tmpf)
        frame_indices = list(range(0, len(video_reader), 1))[:]
        frames = video_reader.get_batch(frame_indices)
        frames = torch.flip(frames, (3,))
        return remotef, tmpf, frames, frame_indices
    except:
        return None


def fetch_videos(video_files, parallel=4):
    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            results = executor.map(fetch_single_video, video_files)
        for res in results:
            yield res
    else:
        for remotef,tmpf in video_files:
            yield fetch_single_video(remotef,tmpf)

def batch_warpper(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i: i+bs]

def cuda_transform(batch_imgs, args, mode='dino'):
    if mode in ('dino', 'blip', 'blip2', 'blip_vqa'):
        if mode == 'blip_vqa':
            size = (480, 480)
        elif mode == 'dino':
            size = (args.det_size, args.det_size)
        elif mode == 'blip' or mode == 'blip2':
            size = (384, 384)
        else:
            size = (364, 364)

        if mode == 'dino':
            mean, std = [0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]
        else:
            mean, std = (0.48145466, 0.4578275,
                         0.40821073), (0.26862954, 0.26130258, 0.27577711)
        batch_imgs = [F.interpolate(
            img, size) for img in batch_imgs]
        batch_imgs = torch.cat(batch_imgs, dim=0)
        batch_imgs = batch_imgs / 255
        batch_imgs = F_t.normalize(batch_imgs, mean, std, inplace=True)
    else:
        raise ValueError
    return batch_imgs

import subprocess
def verbose_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, text=True)

def sync_file(src_file,trg_dir = './tmp/',tmp_name = 'worker0_tmp',
                oss_alias = "aws --endpoint-url=https://tos-s3-cn-shanghai.ivolces.com s3",
                overwrite = False):

    suffix = src_file.split('.')[-1]
    trg_file = os.path.join(trg_dir,f'{tmp_name}.{suffix}')
    
    verbose_cmd(f'{oss_alias} cp {src_file} {trg_file}')
    # os.system(f'/usr/bin/rclone sync {src_file} {trg_file}')
    return src_file,trg_file

def sync_and_untar_files(src_file,trg_dir = './tmp/',
                        oss_alias = "aws --endpoint-url=https://tos-s3-cn-shanghai.ivolces.com s3",
                        overwrite = False):

    tarfname = os.path.split(src_file)[1]
    untar_dir = os.path.join(trg_dir,tarfname.rsplit('.',1)[0])
    trg_file = os.path.join(trg_dir,tarfname)
    
    if not os.path.exists(trg_file) or overwrite:
        os.system(f'{oss_alias} cp {src_file} {trg_dir}')
    if not os.path.exists(trg_dir) or overwrite: 

        os.system(
            f'mkdir {untar_dir};' +
            f"tar --skip-old-files -xf {trg_file} -C {untar_dir};"
        )
    return untar_dir

def worker(args):
    # os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    #NOTE: for no illegal mem access in cuda extensions
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(args.worker_cnt)
    # print(f'{torch.cuda.device_count()}'.center(100,'-'))
    # device ='cuda:0'
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda:0' if not args.device else args.device
        
    # blip2_model = load_model_and_preprocess(
    #         name="blip2", model_type="coco", is_eval=True, device=device)[0]
    # blip2_caption_model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    # load_blip_vqa
    blip_vqa_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    face_ana = load_landmarks_model()
    
    # TODO:
    # faceparsing_model = 
    #  
    args.out_dir = os.path.join(args.out_dir,f'worker{args.worker_cnt}')
    msgpack_writer = AutoSplitMsgPacker(out_dir=args.out_dir, split_size=1 ,overwrite=True)
    condition_out_dir = os.path.join(args.out_dir, 'conditions')

    # pbar = tqdm()
    n_total = len(args.video_files)
    video_files = args.video_files
    if all([f.endswith('.tar') and f.startswith('s3') for f in args.video_files]):
        print('sync tarfiles to local file system')
        def video_files_gen():
            for tarf in args.video_files:
                untar_dir = sync_and_untar_files(tarf)
                for video_f in megfile.smart_glob(os.path.join(untar_dir,'*',"*.mp4")):
                    yield video_f
        video_files = video_files_gen()
    elif all([f.endswith('.mp4') and f.startswith('s3') for f in args.video_files]):
        def video_files_gen():
            for videof in args.video_files:
                yield sync_file(videof, tmp_name= f'{os.path.basename(__file__)}-{args.worker_cnt}')
        video_files = video_files_gen()
    else:
        video_files = [(f,f) for f in video_files]
    
    videos_gen = fetch_videos(
        video_files, parallel=0)
    
    # use_watermark_remover = True
    # if use_watermark_remover:
    #     from tools.watermark_remover.shutterstock import ShutterstockWatermarkRemover
    #     watermark_remover = ShutterstockWatermarkRemover()
    
    total_bar = tqdm(total = n_total,desc = f'worker-{args.worker_cnt:02}-total')
    okbar = tqdm(desc = f'worker-{args.worker_cnt:02}-ok')
    for res in videos_gen:
        if res is None:
            continue
        
        if msgpack_writer.fio is None:
            msgpack_writer.increase_count()
            continue
        total_bar.update(1)
        remotef, tmpf, all_frames, frame_indices = res
        print(remotef,tmpf)
        video_meta = dict()
        video_meta['frames'] = dict()
        video_meta["video_file"] = remotef
        
        video_fname = os.path.split(
            video_meta['video_file'])[-1].rsplit('.', 1)[0]

        # remove watermark
        # if use_watermark_remover:
        #     watermark_free_all_frames = []
        #     flags = []
        #     for k,frame in enumerate(all_frames):
        #         new_frame,flag = watermark_remover(frame,k)
        #         watermark_free_all_frames.append(
        #             new_frame
        #         )
        #         flags.append(flag)
            
        #     video_meta['kf_indices'] = [item for i,item in enumerate(video_meta['kf_indices']) if flags[i]]
        #     all_frames = [item for i,item in enumerate(watermark_free_all_frames) if flags[i]]

        for batch_frame_idxs, batch_frames in zip(
            batch_warpper(frame_indices, args.batch_size),
            batch_warpper(all_frames, args.batch_size)
        ):

            batch_imgs_tensor = batch_frames.permute(0, 3, 1, 2)
            batch_imgs_tensor = [img.float().to(args.device)[None].permute(0, 3, 1, 2) for img in batch_frames]

            # TODO：BLIP标签需要考虑用法：先crop人脸再打标签
            # 还需要考虑celebv这种已经有标签的情况
            # batch_img_for_blip2 = cuda_transform(
            #     batch_imgs_tensor, args, 'blip2')
            # batch_img_for_blip2 = vis_processors["eval"](batch_imgs_tensor).unsqueeze(0).to(device)

            # start = time.time()
            # with torch.cuda.amp.autocast(), torch.no_grad():
            #     # batch_gen_caption = blip2_caption_model.generate(
            #     #     {'image':
            #     #         batch_img_for_blip2
            #     #      }
            #     # )
            #     batch_img_for_blip = cuda_transform(batch_imgs_tensor, args, 'blip_vqa')
            #     question_1 = txt_processors["eval"]("what emotion in the person's face?")
            #     # question_2 = txt_processors["eval"]("What action in the person's face?")
            #     # question_3 = txt_processors["eval"]("What the person looks like?")
            #     question_batch = [question_1] * batch_img_for_blip.shape[0]
            #     batch_gen_caption = gen_blip_vqa_captions(vis_processors, txt_processors, blip_vqa_model, question_batch, batch_img_for_blip)
            #     batch_gen_caption = ["The person's emotion is " + x for x in batch_gen_caption]
            # end = time.time()
            # print('Blip2 Time: {:.6f}s.'.format(end - start))

            
            for k in range(len(batch_frames)):
                frame_idx = batch_frame_idxs[k]
                ori_h, ori_w = batch_imgs_tensor[k].shape[-2:]
                
                # face detection

                # face landmarks
                start = time.time()
                data_faces, img_with_ldmks = gen_landmarks(batch_frames[k].numpy(), face_ana)
                end = time.time()
                print('gen landmark per frame Time: {:.6f}s.'.format(end - start))
                
                # TODO
                # face parsings

                img_f = os.path.join(
                    condition_out_dir, video_fname, 'frames', f'{frame_idx}.png')
                dump_img(batch_frames[k].numpy(), img_f)

                ldmk_f = os.path.join(
                    condition_out_dir, video_fname, 'landmarks', f'{frame_idx}.pkl')
                with smart_open(ldmk_f, 'wb') as file:
                    pickle.dump(data_faces, file)

                if DEBUG:
                    img_with_ldmks_f = os.path.join(condition_out_dir, video_fname, 'img_with_ldmks', f'{frame_idx}.png')
                    dump_img(img_with_ldmks, img_with_ldmks_f)

                # blip vqa + landmark
                start = time.time()
                if len(data_faces) == 0:
                    batch_gen_caption = ['There is no face in the photo']
                else:
                    with torch.cuda.amp.autocast(), torch.no_grad():
                        x1,y1,x2,y2 = data_faces[0]['bbox']
                        x1 = max(int(x1), 0)
                        y1 = max(int(y1), 0)
                        x2 = min(int(x2), ori_w)
                        y2 = min(int(y2), ori_h)
                        face_image = batch_imgs_tensor[k][...,y1:y2,x1:x2][None]
                        face_image = cuda_transform(face_image, args, 'blip_vqa')
                        question_1 = txt_processors["eval"]("what emotion in the person's face?")
                        # question_2 = txt_processors["eval"]("What action in the person's face?")
                        # question_3 = txt_processors["eval"]("What the person looks like?")
                        # question_batch = [question_1] * batch_img_for_blip.shape[0]
                        batch_gen_caption = gen_blip_vqa_captions(vis_processors, txt_processors, blip_vqa_model, question_1, face_image)
                    batch_gen_caption = ["The person's emotion is " + x for x in batch_gen_caption]
                    end = time.time()
                    print('Blip2 Time: {:.6f}s.'.format(end - start))

                    gender = 'male' if data_faces[0]['gender'] == 0 else 'female'
                    age = data_faces[0]['age']
                    caption = batch_gen_caption[0].replace('person', f'{age} years old {gender}')
                print(caption)
                frame_meta = {
                    'caption': caption,
                    'img': img_f,
                    'landmarks': ldmk_f,
                    # 'faceparsing': ,
                }
                video_meta['frames'][frame_idx] = frame_meta
        video_meta['num_frames'] = len(all_frames)
            
        # if DEBUG:
        #     os.makedirs(f'./tmp/{video_fname}/',exist_ok = True)
        #     for idx, meta in video_meta['frames'].items():
        #         cv2.imwrite(
        #             os.path.join(f'./tmp/{video_fname}/{idx}.png'),
        #             np.array(meta['landmark'])
        #         )

        msgpack_writer.write(video_meta)
        okbar.update(1)

    # if DEBUG is not None and int(DEBUG) == 2 and i > args.batch_size * 2:
    #     break
    msgpack_writer.close()


# TODO: fp16 infer， trt, tokenize, load_tar


@functools.lru_cache(maxsize=1)
def get_current_time():
    now = datetime.now()
    now = now.strftime("%Y%m%d-%H%M%S")
    return now

def get_replica_worker_index():
    
    hostname = os.environ.get("HOSTNAME")
    return int(hostname.split('-')[-1])


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate caption, landmark, faceparsing", add_help=True)
    # TODO: trt models readme
    parser.add_argument("--process-way",type = str, choices = ['face','default'],default = 'default')
    parser.add_argument("--input",type = str, default = './test_videos/*.mp4')
    parser.add_argument("--labels",type = str, default = '')
    parser.add_argument("--use_trt", action="store_true")
    parser.add_argument("--batch_size", type=int,
                        default=256)
    parser.add_argument("--cpu-only", action="store_true",
                        help="running on cpu only!, default=False")
    parser.add_argument("--out_dir", "-n", type=str,
                        required=False, help="nori path")
    parser.add_argument("--max_text_len",  type=int, default=32,
                        required=False, help="nori path")
    parser.add_argument("--n_debug",  type=int, default=-1,
                        required=False, help="nori path")
    parser.add_argument("--total_replica",  type=int, default=1,
                        required=False, help="nori path")
    args = parser.parse_args()
    args.total_replica = int(os.environ.get("MLP_WORKER_NUM",1))
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.worker_cnt = 0
    
        
    cache_decorator = local_file_cache_func(
        './cache_tmp'
    )(megfile.smart_glob)
    
    
    if DEBUG or torch.cuda.device_count() == 1:
        args.video_files = cache_decorator(args.input
        )
        if args.total_replica > 1:
            args.worker_cnt = get_replica_worker_index()
            step = len(args.video_files) // args.total_replica
            
            args.video_files = args.video_files[int(args.worker_cnt*step) : (int(args.worker_cnt*step) + step) if args.worker_cnt < args.total_replica-1 else -1]
            print(f'The current worker {args.worker_cnt} jobs : {len(args.video_files)}')
        print('total_replica, worker_cnt', args.total_replica, args.worker_cnt)
        worker(args)
    else:
        args.video_files = cache_decorator(args.input)
        print(f'Video : {len(args.video_files)}')
        mp.set_start_method('spawn')

        n_parallel = torch.cuda.device_count()
        job_list = [ deepcopy(args) for _ in range(n_parallel)]

        n_total = len(args.video_files)
        step = n_total//n_parallel 
        proc_list = []
        for i,job in enumerate(job_list):
            job.worker_cnt = i
            job.device = f"cuda:{i}"
            job.video_files = args.video_files[int(i*step) : int(i*step) + step if i < n_parallel-1 else -1]
            print(f'Dispatch {len(job.video_files)} jobs to {job.device}')
            proc = mp.Process(
                target=worker,args = (job,)
            )
            proc.start()
            proc_list.append(proc)
        
        for p in proc_list:
            p.join()
        
 
