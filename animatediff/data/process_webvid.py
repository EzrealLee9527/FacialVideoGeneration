# from fastsam import FastSAM, FastSAMPrompt
import settings
from data_fetcher import dump_img, load_img
from collections import defaultdict
from tools.tracker import Sort
from data_fetcher import AutoSplitMsgPacker, split_nouns
from faster_models import ModelTimer
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from tools.keyframe_detector import NaiveKFDector
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
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image, ImageDraw, ImageFont
import orjson
import ripplit as rp
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from line_profiler import LineProfiler
import logging
import warnings

# Filter out specific warning category or message
warnings.filterwarnings("ignore", category=Warning)

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Writes to console
# logger.setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('megfile').setLevel(logging.ERROR)
logging.getLogger('megfile.s3_path').setLevel(logging.ERROR)
# 下载太慢 建议rclone sync -P  brainpp-oss:weisiyuan-dev/nltk_data ~/nltk_data
# 下载NLTK停用词
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
DEBUG = os.environ.get('DEBUG')
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
except:
    print("Fail to set tf32 computation")


def load_fastsam(model_path='/data/ckpts/fast-sam/FastSAM-x.pt'):
    model = FastSAM(model_path)
    return model


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = torch.Tensor(tgt["boxes"])
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):

    if megfile.smart_exists(image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image
        ori_shape = (image_pil.height, image_pil.width)
    else:
        image = imdecode(Fetcher().get(image_path))[:, :, ::-1]
        ori_shape = (image.shape[0], image.shape[1])
        image_pil = Image.fromarray(image)

    transform = T.Compose(
        [
            T.RandomResize([(800)], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image, ori_shape


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"

    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


COLORS_WORDS = ['red', 'blue', 'green', 'yellow', 'purple',
                'orange', 'pink', 'brown', 'black', 'white', 'gray']
BLACK_WORDS = [
    'group', 'front', 'back',
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
    words = [word for (word, pos) in nltk.pos_tag(words)
             if is_noun(pos) and word.lower() not in BLACK_WORDS]

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


@lru_cache(maxsize=4)
def load_tar(p):

    return tarfile.open(fileobj=megfile.smart_open(p, 'rb'))


def load_ckpt(ckp_path):
    with megfile.smart_open(ckp_path, 'rb') as fobj:
        return torch.load(fobj)


def load_model_tokenizer(args, config_file, checkpoint_path, device, cpu_only=False, use_trt=True):
    model = load_model(config_file, checkpoint_path,
                       cpu_only=cpu_only)

    model = model.to(device)

    if use_trt:
        from faster_models import get_trt_swin_det
        model.backbone[0] = get_trt_swin_det(
            bs=args.batch_size, inp=args.det_size)

    tokenlizer = model.tokenizer
    return model, tokenlizer


def load_blip_caption(device):

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    return model, vis_processors


def cuda_transform(batch_imgs, args, mode='dino'):

    if mode in ('dino', 'blip', 'blip2'):

        if mode == 'dino':
            size = (args.det_size, args.det_size)
        elif mode == 'blip':
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


def resize_box(box, src_shape, trg_shape):

    scale = (trg_shape[0]/src_shape[0], trg_shape[1]/src_shape[1])
    return [
        box[0] * scale[1],
        box[1] * scale[0],
        box[2] * scale[1],
        box[3] * scale[0]
    ]


def load_mobilesam(model_type="vit_t",
                   sam_checkpoint="/data/users/weisiyuan/ckpts/mobile_sam.pt",
                   device='cuda', img_size=1024):

    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    mobile_sam = sam_model_registry[model_type](
        checkpoint=sam_checkpoint, img_size=img_size)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)
    return predictor


def dump_packed_mask(m, f):
    with megfile.smart_open(f, 'wb') as fio:
        shape = m.shape
        packed_mask = np.packbits(m.reshape(-1))
        pickle.dump([shape, packed_mask], fio)


def load_packed_mask(f, dtype='uint8'):

    with megfile.smart_open(f, 'rb') as fio:
        shape, packed_mask = pickle.loads(fio.read())
        unpack_mask = np.unpackbits(packed_mask)
        length = int(reduce(lambda a, b: a * b, shape))
        mask = unpack_mask[:length].reshape(*shape)
    return mask


def fetch_single_video(remotef,tmpf,thre=0.6, trg_shape=(224, 224)):
    try:
        indices, kf_list, n_total = NaiveKFDector(
                    tmpf, thre, trg_shape=trg_shape).process(decoder='opencv', bgr2rgb=True)
        # os.system(f'rm {os.path.abspath(tmpf)}')
        return remotef,tmpf, indices, kf_list, n_total
    except:
        return None


def fetch_video_key_frames(video_files, parallel=4):


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


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = torch.Tensor(tgt["boxes"])
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font, spacing=2)
        else:
            w, h = draw.textsize(str(label), font, spacing=2)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def plot_frame_info(meta):

    img = load_img(meta['img'])
    meta['size'] = img.shape[:2]
    meta['labels'] = [
        f"{trk_id}:{label}"
        for trk_id, label in zip(meta['track_ids'], meta['labels'])
    ]
    canvas = plot_boxes_to_image(Image.fromarray(img), meta)[0]
    return canvas

from tools.tracker.sort import reset_tracker_count

import subprocess
def verbose_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, text=True)




def sync_file(src_file,trg_dir = '/data/users/weisiyuan/tmp/',tmp_name = 'worker0_tmp',
                oss_alias = "aws --endpoint-url=https://tos-s3-cn-shanghai.ivolces.com s3",
                overwrite = False):

    suffix = src_file.split('.')[-1]
    trg_file = os.path.join(trg_dir,f'{tmp_name}.{suffix}')
    
    verbose_cmd(f'{oss_alias} cp {src_file} {trg_file}')
    # os.system(f'/usr/bin/rclone sync {src_file} {trg_file}')
    return src_file,trg_file


def sync_and_untar_files(src_file,trg_dir = '/data/users/weisiyuan/tmp/',
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

import cv2
# TODO: 需要找一些条件过滤掉一些低质量视频


def set_env_for_multi():
    cv2.setNumThreads(8)
    os.environ["OMP_NUM_THREADS"] = "8"


def worker(args):
    # os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    #NOTE: for no illegal mem access in cuda extensions
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(args.worker_cnt)
    # print(f'{torch.cuda.device_count()}'.center(100,'-'))
    # device ='cuda:0'
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda:0' if not args.device else args.device
    
        
    
    sam_model = load_mobilesam(img_size=args.seg_size,device = args.device)
    
    blip2_model = load_model_and_preprocess(
            name="blip2", model_type="coco", is_eval=True, device=device)[0]
    args.out_dir = os.path.join(args.out_dir,f'worker{args.worker_cnt}')
    msgpack_writer = AutoSplitMsgPacker(out_dir=args.out_dir, split_size=1 ,overwrite=True)
    mask_out_dir = os.path.join(args.out_dir, 'masks')

    model, tokenlizer = load_model_tokenizer(args,
                                             config_file, checkpoint_path, device, args.cpu_only, args.use_trt)

    # pbar = tqdm()
    # timer = ModelTimer(pbar)

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
    
    video_kfs_gen = fetch_video_key_frames(
        video_files, parallel=0)
    
    use_watermark_remover = True

    if use_watermark_remover:
        from tools.watermark_remover.shutterstock import ShutterstockWatermarkRemover
        watermark_remover = ShutterstockWatermarkRemover()
    
    total_bar = tqdm(total = n_total,desc = f'worker-{args.worker_cnt:02}-total')
    okbar = tqdm(desc = f'worker-{args.worker_cnt:02}-ok')
    for res in video_kfs_gen:
        if res is None:
            continue
        
        if msgpack_writer.fio is None:
            msgpack_writer.increase_count()
            continue
        total_bar.update(1)
        remotef,tmpf, indices, kf_list, n_total = res
        # print(remotef,tmpf)
        video_meta = dict()
        video_meta['frames'] = dict()
        video_meta["video_file"], video_meta['kf_indices'],kfs, video_meta['total_frames'] = remotef,indices, kf_list, n_total
        video_meta['kf_indices'] = [ int(idx) for idx in video_meta['kf_indices']]
        video_fname = os.path.split(
            video_meta['video_file'])[-1].rsplit('.', 1)[0]
        # define tracker
        trk_id2frame_ids = defaultdict(list)
        reset_tracker_count()
        mot_tracker = Sort(max_age=video_meta['total_frames'],
                           min_hits=2,  # Maximum number of frames to keep alive a track without associated detections, when frame_count > min_hits
                           iou_threshold=0.3)  # Minimum IOU for match.

        # remove watermark
        if use_watermark_remover:
            watermark_free_kfs = []
            flags = []
            for k,frame in enumerate(kfs):
                new_frame,flag = watermark_remover(frame,k)
                watermark_free_kfs.append(
                    new_frame
                )
                flags.append(flag)
            
            video_meta['kf_indices'] = [item for i,item in enumerate(video_meta['kf_indices']) if flags[i]]
            kfs = [item for i,item in enumerate(watermark_free_kfs) if flags[i]]
        if not len(video_meta['kf_indices']):
            continue
        for batch_frame_idxs, batch_frames in zip(
            batch_warpper(video_meta['kf_indices'], args.batch_size),
            batch_warpper(kfs, args.batch_size)
        ):

            batch_imgs_tensor = [torch.from_numpy(img.copy()).float().to(args.device)[None].permute(0, 3, 1, 2) for img in batch_frames]

            # with timer.measure('blip2_inp_transform'):
            batch_img_for_blip2 = cuda_transform(
                batch_imgs_tensor, args, 'blip2')

            with torch.cuda.amp.autocast(), torch.no_grad():
                batch_gen_caption = blip2_model.generate(
                    {'image':
                        batch_img_for_blip2
                     }
                )
            # 如果没有提供labels则使用model generate
            batch_gen_nouns = [
                split_nouns(cap, min_caption_len=1)
                for cap in batch_gen_caption
            ] if not args.labels else [args.labels for _ in range(len(batch_gen_caption))]
            
            # with timer.measure('dino_inp_transform'):
            batch_images = cuda_transform(batch_imgs_tensor, args, 'dino')
            batch_images = nested_tensor_from_tensor_list(batch_images)
            # with timer.measure('encode_img'):
            with torch.cuda.amp.autocast(), torch.no_grad():
                batch_feats, batch_poss = model.encode_image(batch_images)

            with torch.no_grad(), torch.cuda.amp.autocast():
                batch_text_dict = model.encode_text(
                    batch_gen_nouns,device)

            for k in range(len(batch_images)):

                if batch_gen_nouns[k] is None:
                    continue
                frame_idx = batch_frame_idxs[k]
                ori_h, ori_w = batch_imgs_tensor[k].shape[-2:]
                with torch.no_grad(), torch.cuda.amp.autocast():
                    sam_model.set_image(
                        batch_imgs_tensor[k]
                    )

                image, feats, poss = batch_images[k:k + 1], [f[k:k+1]
                                                             for f in batch_feats], [p[k:k+1] for p in batch_poss]
                # image = nested_tensor_from_tensor_list(image)
                all_box = []
                all_pred = []
                all_score = []

                sub_caption_list, text_dict_list = [
                    batch_gen_nouns[k]], [{key: v[k:k+1] for key, v in batch_text_dict.items()}]
                sub_tokenized_list = [model.tokenizer(
                    sub_caption) for sub_caption in sub_caption_list]
                for text_dict, sub_tokenized in zip(text_dict_list, sub_tokenized_list):

                    text_dict = {k: v.clone() for k, v in text_dict.items()}
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        outputs = model.bind_images_text(
                            text_dict, image, feats, poss)
                    logits = outputs["pred_logits"].cpu().sigmoid()[
                        0]  # (nq, 256)
                    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
                    logits_filt = logits.clone()
                    boxes_filt = boxes.clone()
                    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
                    logits_filt = logits_filt[filt_mask]  # num_filt, 256
                    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4\

                    boxes_filt = boxes_filt.cpu().numpy().tolist()
                    all_box += boxes_filt

                # with timer.measure('fastsam_prompt_process'):
                boxes_xyxy = []

                # TODO: 可能需要提前过滤下bbox
                for logit, box in zip(logits_filt, boxes_filt):

                    pred_phrase = get_phrases_from_posmap(
                        logit > text_threshold, sub_tokenized, tokenlizer)
                    all_pred.append(pred_phrase)
                    all_score.append(logit.max().item())

                    # convert,cx,cy,w,h to x0y0x1y1
                    resized_box = resize_box(
                        box, (1, 1), (ori_h, ori_w))
                    resized_box = np.array(resized_box)
                    resized_box[:2] -= resized_box[2:] / 2
                    resized_box[2:] += resized_box[:2]
                    resized_box = resized_box.tolist()
                    boxes_xyxy.append(resized_box)

                # dump masks by packing it to a uint8 arr
                if len(boxes_xyxy):
                    # update tracker
                    # TODO: tracker可能无法处理挑帧的情况
                    _, track_ids = mot_tracker.update(np.array(boxes_xyxy))
                    for trk_id in track_ids:
                        if trk_id is not None:
                            trk_id2frame_ids[trk_id].append(frame_idx)

                    masks = sam_model.predict(
                        box=np.array(boxes_xyxy))[0]
                    masks_f = os.path.join(
                        mask_out_dir, video_fname,'masks', f'{frame_idx}.pkl')
                    img_f = os.path.join(
                        mask_out_dir, video_fname,'frames', f'{frame_idx}.png')
                    masks = np.transpose(masks, (0, 2, 3, 1))[:, :, :, -1]
                    dump_packed_mask(masks, masks_f)
                    dump_img(batch_frames[k],img_f)
                    # 总是出现group people ,group men, people front，man beard这样的
                    frame_meta = {
                        'track_ids': [ int(i) for i in track_ids],
                        'boxes': all_box,
                        'labels': all_pred,
                        'scores': all_score,
                        'caption': batch_gen_caption[k],
                        'masks': masks_f,
                        'img': img_f,
                    }
                    video_meta['frames'][frame_idx] = frame_meta

        if DEBUG:
            os.makedirs(f'/data/users/weisiyuan/tmp/rp_outputs/{video_fname}/',exist_ok = True)
            for idx, meta in video_meta['frames'].items():
                # rp.imshow(f"{idx} : {meta['caption']}", plot_frame_info(meta))
                cv2.imwrite(
                    os.path.join(f'/data/users/weisiyuan/tmp/rp_outputs/{video_fname}/{idx}.png'),
                    np.array(plot_frame_info(meta))
                )

            # rp.waitKey(0)
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

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str,
                        default='./groundingdino/config/GroundingDINO_SwinB_cfg.py', help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default='/data/users/weisiyuan/ckpts/groundingdino_swinb_cogcoor.pth', help="path to checkpoint file"
    )
    # TODO: trt models readme
    parser.add_argument("--process-way",type = str, choices = ['face','default'],default = 'default')
    parser.add_argument("--input",type = str, default = '/data/users/weisiyuan/dataset/webvid-10M-train/*/*.mp4')
    parser.add_argument("--labels",type = str, default = '')
    parser.add_argument("--includes",type = str, 
                        default = 'person,girl,boy,child,man,woman,women,'
                        +'cat,dog,car,bicycle,painting,drawing')
    parser.add_argument("--use_trt", action="store_true")
    parser.add_argument("--box_threshold", type=float,
                        default=0.3, help="box threshold")
    parser.add_argument("--batch_size", type=int,
                        default=16)
    parser.add_argument("--det_size", type=int,
                        default=384)
    parser.add_argument("--seg_size", type=int,
                        default=1024)
    parser.add_argument("--min_size", type=int,
                        default=512)
    parser.add_argument("--text_threshold", type=float,
                        default=0.25, help="text threshold")
    parser.add_argument("--detect_file_only",
                        action="store_true", help="text threshold")
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
    
    
    from utils import local_file_cache_func
    cache_decorator = local_file_cache_func(
        '/data/user/weisiyuan/tmp'
    )(megfile.smart_glob)
    
    
    if args.includes:
        meta_info_file = '/data/users/weisiyuan/dataset/results_10M_train.csv'
        import pandas as pd
        includes = args.includes.split(',')
        meta_infos = pd.read_csv(meta_info_file)
        
        mask = meta_infos['name'].apply(lambda x: any(kw in x for kw in includes))
        valid_videoids = set(meta_infos[mask]['videoid'].tolist())
        
        
    if DEBUG or torch.cuda.device_count() == 1:
        args.video_files = cache_decorator(args.input
        )
    
        
        if args.includes:
            args.video_files = [
                f for f in args.video_files if int(os.path.split(f)[-1].split('.')[0]) in valid_videoids
            ][:args.n_debug]
            print(f'Video : {len(args.video_files)}')
        if args.total_replica > 1:
            args.worker_cnt = get_replica_worker_index()
            step = len(args.video_files) // args.total_replica
            
            args.video_files = args.video_files[int(args.worker_cnt*step) : (int(args.worker_cnt*step) + step) if args.worker_cnt < args.total_replica-1 else -1]
            print(f'The current worker {args.worker_cnt} jobs : {len(args.video_files)}')
        worker(args)
    else:
        args.video_files = cache_decorator(args.input)
        print(f'Video : {len(args.video_files)}')
        if args.includes:
            args.video_files = [
                f for f in args.video_files if int(os.path.split(f)[-1].split('.')[0]) in valid_videoids
            ]
            print(f'Video : {len(args.video_files)}')
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')

        from copy import deepcopy
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
        
 
