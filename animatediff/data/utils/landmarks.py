# Face alignment and crop demo
# Uses MTCNN, FaceBoxes or Retinaface as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Retinaface+MobileFaceNet gives the best peformance
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021
from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
import sys
sys.path.append('/work00/AnimateDiff-adapter')
from pytorch_face_landmark.common.utils import BBox, drawLandmark,drawLandmark_multiple, drawLandmark_multiple_nobox
from pytorch_face_landmark.models.basenet import MobileNet_GDConv
from pytorch_face_landmark.models.pfld_compressed import PFLDInference
from pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
from pytorch_face_landmark.FaceBoxes import FaceBoxes
from pytorch_face_landmark.Retinaface import Retinaface
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_face_landmark.MTCNN import detect_faces
import glob
import time
import imageio
import json
from collections import OrderedDict
from pytorch_face_landmark.utils.align_trans import get_reference_facial_points, warp_and_crop_face
from multiprocessing import Process
import os
from decord import VideoReader, cpu
from functools import lru_cache

@lru_cache(maxsize=None)
def get_model():
    map_location='cpu'
    checkpoint = torch.load('/work00/AnimateDiff-adapter/pytorch_face_landmark/checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)  
    model = MobileFaceNet([112, 112],136)   
    model.load_state_dict(checkpoint['state_dict']) 
    return model


def get_landmarks(frames):
    # print(f"Processing landmarks")  
    model = get_model() 
    model = model.eval()
    backbone = 'MobileFaceNet'
    mean = np.asarray([ 0.485, 0.456, 0.406 ])
    std = np.asarray([ 0.229, 0.224, 0.225 ])
    detector = 'Retinaface'
    out_size = 112 
    crop_size= 112
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    
    start = time.time()
    result = OrderedDict({"frames": []})
    ret_landmarks = []

    for frame_idx, img in enumerate(frames):
        frame_info = {
            "frame_id": frame_idx,
            "faces": []
        }
        

        height,width,_=img.shape
        
        if detector=='MTCNN':
            # perform face detection using MTCNN
            image = Image.fromarray(img)
            faces, landmarks = detect_faces(image)
        elif detector=='FaceBoxes':
            face_boxes = FaceBoxes()
            faces = face_boxes(img)
        elif detector=='Retinaface':
            m_start = time.time()
            retinaface=Retinaface.Retinaface()    
            faces = retinaface(img)     
            m_end = time.time()    
            # print('model', m_end-m_start)   
        else:
            print('Error: not suppored detector')        
        ratio=0
        if len(faces)==0:
            print('NO face is detected!')
            continue
        # print('faces', len(faces))
        for k, face in enumerate(faces): 
            if face[4]<0.9: # remove low confidence detection
                continue
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h])*1.2)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face/255.0
            if backbone=='MobileNet':
                test_face = (test_face-mean)/std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input= torch.autograd.Variable(input)
            
            if backbone=='MobileFaceNet':
                landmark = model(input)[0].cpu().data.numpy()
            else:
                landmark = model(input).cpu().data.numpy()
            
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            # landmark = np.array(landmark,np.int32)

            draw_img = np.zeros_like(img)
            img = drawLandmark_multiple_nobox(draw_img, landmark)
            # img = drawLandmark_only(draw_img, landmark)
            # crop and aligned the face
            # print('crop and aligned the face')
            # lefteye_x=0
            # lefteye_y=0
            # for i in range(36,42):
            #     lefteye_x+=landmark[i][0]
            #     lefteye_y+=landmark[i][1]
            # lefteye_x=lefteye_x/6
            # lefteye_y=lefteye_y/6
            # lefteye=[lefteye_x,lefteye_y]

            # righteye_x=0
            # righteye_y=0
            # for i in range(42,48):
            #     righteye_x+=landmark[i][0]
            #     righteye_y+=landmark[i][1]
            # righteye_x=righteye_x/6
            # righteye_y=righteye_y/6
            # righteye=[righteye_x,righteye_y]  

            # nose=landmark[33]
            # leftmouth=landmark[48]
            # rightmouth=landmark[54]
            # facial5points=[righteye,lefteye,nose,rightmouth,leftmouth]
            # warped_face = warp_and_crop_face(np.array(org_img), facial5points, reference, crop_size=(crop_size, crop_size))
            # img_warped = Image.fromarray(warped_face)

            face_info = {
                "face_idx": k,
                "landmarks": landmark.tolist()
            }
            frame_info["faces"].append(face_info)

        result["frames"].append(frame_info)        
        
        # save the landmark detections 
        # print(os.path.join('/work00/AnimateDiff-adapter/pytorch_face_landmark/results','test_dataset_img.png'))
        # cv2.imwrite(os.path.join('/work00/AnimateDiff-adapter/pytorch_face_landmark/results',f'{frame_idx}_test_dataset_img.png'),frames[frame_idx])
        # ldmks_on_img = drawLandmark_multiple(img, new_bbox, landmark)
        # cv2.imwrite(os.path.join('/work00/AnimateDiff-adapter/pytorch_face_landmark/results',f'{frame_idx}_test_dataset_ldmks.png'),ldmks_on_img)
        # cv2.imwrite(os.path.join('/work00/AnimateDiff-adapter/pytorch_face_landmark/results',f'{frame_idx}_test_dataset_ldmks1.png'),img)
        # frame_idx += 1
        # print('frame_idx', frame_idx)

        # print('draw_img', frames[0].shape, img.shape)

        ret_landmarks.append(img)

    end = time.time()
    # print('Time: {:.6f}s.'.format(end - start))

    
    return ret_landmarks, result


if __name__ == "__main__":
    

    meta_path = '/dataset00/Videos/CelebV-Text/descripitions/happy_filelist.txt'
    # meta_path = '/dataset00/Videos/CelebV-Text/descripitions/remains_happy_filelist.txt'
    with open(meta_path, 'r') as txt_file:
        lines = txt_file.readlines()
        video_files = ['/dataset00/Videos/CelebV-Text/videos/celebvtext_6/'+x.strip() for x in lines]

    video_reader = VideoReader(video_files[0], ctx=cpu(0))
    frame_indices = list(range(8))
    frames = video_reader.get_batch(frame_indices).asnumpy()
    # print(frames.shape, type(frames), frames.max(),frames.min())
    res = get_landmarks(frames)
    # print('res', res)

    
    
