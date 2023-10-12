import numpy as np
import sys
sys.path.append('/work00/FacialVideoGeneration/')
from animatediff.data.face_model import FaceAnalysis
import imageio
import cv2
import numpy as np

def draw_landmarks(image, landmarks, radius=2, color=(0, 255, 0)):
    for x, y in landmarks:
        cv2.circle(image, (int(x), int(y)), radius, color, thickness=-1)
    return image
def load_landmarks_model():
    face_ana = FaceAnalysis()
    face_ana.prepare(ctx_id=0, det_size=(640, 640))
    return face_ana

def gen_landmarks(img, face_ana, draw=False):
    res = face_ana.get(img, 1)
    if draw:
        img = face_ana.draw_on(img, res)
    # res[0] : dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
    return [dict(x) for x in res], img

def calculate_mean_landmark(landmarks):
    return np.mean(landmarks, axis=0)

def transform_sequence(a_sequence, a_mean, b_mean, scale_factor):
    transformed_sequence = []
    for a_frame in a_sequence:
        print('debug', a_frame.shape, a_mean, b_mean)
        translated_frame = a_frame - a_mean + b_mean        
        scaled_frame = translated_frame * scale_factor
        transformed_sequence.append(scaled_frame)
    return transformed_sequence

template_video_path = '/work00/FacialVideoGeneration/animatediff/data/gen_ldmk_sqc/datas/6-a-person-is-smiling,-High-detailed,-High-precision,-Hyper-quality,.gif'
gen_ref_frame_path = '/work00/FacialVideoGeneration/animatediff/data/gen_ldmk_sqc/datas/38_template.png'

face_ana = load_landmarks_model()
reader = imageio.mimread(template_video_path) 

template_orthofacial_frame = reader[-1]

gen_ref_frame = cv2.imread(gen_ref_frame_path)
data_faces, img_with_ldmks = gen_landmarks(gen_ref_frame, face_ana)
gen_ref_ldmk = data_faces[0]['landmark_2d_106']

# 计算每一帧的平移和缩放
# 平移：关键帧（正脸）和
# 缩放：关键帧和ref帧bbox算缩放

video_size = (640, 640)  
fps = 30
output_video_file = "template_landmarks_sequence.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, video_size)    

template_landmarks_sequence = []
for f_idx in range(len(reader)):
    frame = reader[f_idx][:,:,:3]
    data_faces, img_with_ldmks = gen_landmarks(frame, face_ana)
    ldmk = data_faces[0]['landmark_2d_106']
    template_landmarks_sequence.append(ldmk)

    image = np.zeros((video_size[1], video_size[0], 3), dtype=np.uint8)
    image_with_landmarks = draw_landmarks(image, ldmk) 
    video_writer.write(image_with_landmarks)
video_writer.release()
print('template_landmarks_sequence', len(template_landmarks_sequence), template_landmarks_sequence[0].shape)



# 计算A人脸关键点序列的平均位置
template_mean_landmark = calculate_mean_landmark(template_landmarks_sequence)

# 计算B人脸单帧关键点的平均位置
gen_ref_mean_landmark = calculate_mean_landmark(gen_ref_ldmk)

# 估算A与B之间的缩放因子，这里假定为1.0
# 实际应用中可以根据特征（如眼睛、鼻子或嘴巴的距离）计算缩放因子
scale_factor = 1.0

# 转换A人脸关键点序列以生成B人脸关键点序列
gen_landmarks_sequence = transform_sequence(template_landmarks_sequence, template_mean_landmark, gen_ref_mean_landmark, scale_factor)
print('gen_landmarks_sequence', len(gen_landmarks_sequence), gen_landmarks_sequence[0].shape)
# print(gen_landmarks_sequence)

video_size = (640, 640)  
fps = 30
output_video_file = "gen_landmarks_sequence.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, video_size)
for frame_landmarks in gen_landmarks_sequence:
    image = np.zeros((video_size[1], video_size[0], 3), dtype=np.uint8)
    image_with_landmarks = draw_landmarks(image, frame_landmarks) 
    video_writer.write(image_with_landmarks)
video_writer.release()
