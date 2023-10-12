import numpy as np
import sys
sys.path.append('/work00/FacialVideoGeneration/')
from animatediff.data.face_model import FaceAnalysis
import imageio
import cv2
import numpy as np
from ldmks2ldmks import load_landmarks_model, gen_landmarks, draw_landmarks
from skimage import transform
from PIL import Image
import os
from animatediff.data.datasets import generate_gaussian_response
'''
if f_idx == 0: 
        # 轮廓
        fusion_landmarks = list(paste_landmarks[:33]) #+ list(paste_landmarks[-33:])
    if f_idx == 1: 
        # 左眼
        fusion_landmarks = list(paste_landmarks[33:43])
    if f_idx == 2: 
        # 左眉
        fusion_landmarks = list(paste_landmarks[43:52])
    if f_idx == 3: 
        # 嘴巴
        fusion_landmarks = list(paste_landmarks[52:72])
    if f_idx == 4: 
        # 鼻子
        fusion_landmarks = list(paste_landmarks[72:87])
    if f_idx == 5: 
        # 右眼
        fusion_landmarks = list(paste_landmarks[87:97])
    if f_idx == 6: 
        # 右眉
        fusion_landmarks = list(paste_landmarks[97:])
2D 106 landmarks index:
脸部轮廓（从右侧耳际沿下巴到左侧耳际）：索引 0 - 32 (共33个点)
左眼（左到右）：索引 33 - 42 (共10个点)
左眉毛（左到右）：索引 43 - 51 (共9个点)
左眉毛（左到右）：索引 52 - 71 (共20个点)
鼻子（上到下）：索引 72 - 86 (共15个点)
右眼（左到右）：索引 87 - 96 (共10个点)
右眉毛（左到右）：索引 97 - 105 (共9个点)

嘴巴外轮廓（顺时针）：索引 
嘴巴内轮廓（顺时针）：索引 
'''
def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box):
    '''
    Inputs:
        Source_image            原图像；
        Source_image_mask       原图像人脸的mask比例；
        Target_image            目标模板图像；
        Source_Five_Point       原图像五个人脸关键点；
        Target_Five_Point       目标图像五个人脸关键点；
        Source_box              原图像人脸的坐标；
    
    Outputs:
        output                  贴脸后的人像
    '''
    Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
    Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

    Crop_Source_image                       = Source_image.crop(np.int32(Source_box))
    Crop_Source_image_mask                  = Source_image_mask.crop(np.int32(Source_box))
    Source_Five_Point, Target_Five_Point    = np.array(Source_Five_Point), np.array(Target_Five_Point)

    tform = transform.SimilarityTransform()
    # 程序直接估算出转换矩阵M
    tform.estimate(Source_Five_Point, Target_Five_Point)
    M = tform.params[0:2, :]

    warped      = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

    mask        = np.float32(warped_mask == 0)
    output      = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    return output

# template_video_path = '/work00/FacialVideoGeneration/animatediff/data/gen_ldmk_sqc/datas/6-a-person-is-smiling,-High-detailed,-High-precision,-Hyper-quality,.gif'
# gen_ref_frame_path = '/work00/FacialVideoGeneration/animatediff/data/gen_ldmk_sqc/datas/38_template.png'

template_video_path = '/work00/FacialVideoGeneration/animatediff/data/gen_ldmk_sqc/datas/10_The_person_is_happiness_video.mp4'
gen_ref_frame_path = '/work00/FacialVideoGeneration/animatediff/data/gen_ldmk_sqc/datas/38_mj_gen.png'


face_ana = load_landmarks_model()
if template_video_path.endswith('.gif'):
    reader = imageio.mimread(template_video_path)
else:
    reader = []
    cap = cv2.VideoCapture(template_video_path)
    frames_ldmks = []  
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            reader.append(frame)
        else:
            break

template_orthofacial_frame = reader[-1][:,:,:3][:,:,::-1]

gen_ref_frame = cv2.imread(gen_ref_frame_path)
Source_image = Image.fromarray(gen_ref_frame)
gen_ref_frame_faces, img_with_ldmks = gen_landmarks(gen_ref_frame, face_ana)
Source_Five_Point = gen_ref_frame_faces[0]['kps']
Source_box = gen_ref_frame_faces[0]['bbox']
Source_image_mask = Image.fromarray(np.ones_like(Source_image))

video_size = reader[0].shape[:2] 
fps = 30
output_video_file = "crop_and_paste.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, video_size) 
ldmk_output_video_file = template_video_path.replace('.mp4',"_paste_and_fusion.mp4")
ldmk_video_writer = cv2.VideoWriter(ldmk_output_video_file, fourcc, fps, video_size)   

same_ldmk_output_video_file = template_video_path.replace('.mp4',"_same_ldmk.mp4")
same_ldmk_video_writer = cv2.VideoWriter(same_ldmk_output_video_file, fourcc, fps, video_size)   
same_ldmk_gaussian_response = generate_gaussian_response(list(video_size), gen_ref_frame_faces[0]['landmark_2d_106'], sigma=1) * 255

paste_ldmk_output_video_file = template_video_path.replace('.mp4',"_paste.mp4")
paste_ldmk_video_writer = cv2.VideoWriter(paste_ldmk_output_video_file, fourcc, fps, video_size)   

save_image_dir = 'crop_and_paste'
if not os.path.exists(save_image_dir):
    os.mkdir(save_image_dir)
for f_idx in range(len(reader)):
    frame = reader[f_idx][:,:,:3]
    frame_faces, img_with_ldmks = gen_landmarks(frame, face_ana)
    Target_Five_Point = frame_faces[0]['kps']
    # Target_image = Image.fromarray(frame)
    Target_image = Image.fromarray(np.zeros_like(frame))
    
    output = crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box)
    # draw landmarks
    paste_faces, output_with_ldmks = gen_landmarks(output, face_ana, True)
    # 原始眼睛、鼻子、嘴巴的landmark代替新生成脸的landmark
    fusion_landmarks = []
    frame_landmarks = frame_faces[0]['landmark_2d_106']
    paste_landmarks = paste_faces[0]['landmark_2d_106']
    fusion_landmarks = list(frame_landmarks[33:]) + list(paste_landmarks[:33])
    output_with_fusion_landmarks = draw_landmarks(output, fusion_landmarks) 
    print('output', output.shape)
    cv2.imwrite(f'{save_image_dir}/output_with_fusion_landmarks_{f_idx}.png', output_with_fusion_landmarks)
    video_writer.write(output.astype('uint8'))
    # save_landmarks
    gaussian_response = generate_gaussian_response(list(video_size), fusion_landmarks, sigma=1) * 255
    # save_landmarks = draw_landmarks(np.zeros_like(output), fusion_landmarks, color=(255,255,255)) 
    print('gaussian_response', gaussian_response.shape, gaussian_response.max(), gaussian_response.min())
    ldmk_video_writer.write(np.repeat(gaussian_response.astype('uint8'),3,axis=2))
    same_ldmk_video_writer.write(np.repeat(same_ldmk_gaussian_response.astype('uint8'),3,axis=2))

    # paste
    paste_gaussian_response = generate_gaussian_response(list(video_size), paste_landmarks, sigma=1) * 255
    paste_ldmk_video_writer.write(np.repeat(paste_gaussian_response.astype('uint8'),3,axis=2))
    
video_writer.release()
ldmk_video_writer.release()
same_ldmk_video_writer.release()
paste_ldmk_video_writer.release()