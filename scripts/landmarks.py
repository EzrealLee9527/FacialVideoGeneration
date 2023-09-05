import json
import imageio
import numpy as np
import cv2

def get_landmarks2(video_path, video_length=8, resolution=(512,512)):
    json_path = video_path.replace('gifs', 'landmarks').replace('.gif','.json')
    with open(json_path) as f:
        info = json.load(f)
    reader = imageio.mimread(video_path)
    
    frames_ldmks = []
    frames = []
    for f_idx in range(video_length):
        frame = reader[f_idx]
        width, height =  frame.shape[:2]
        new_width, new_height = resolution
        
        ldmks = info['frames'][f_idx]['faces'][-1]['landmarks']
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

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
        print(cropped_frame.shape, (new_height, new_width))
        resized_frame = cv2.resize(cropped_frame, (new_height, new_width))
        
        frame_ldmks = np.zeros((new_height, new_width,1))
        for x,y in ldmks:
            xc = x - crop_x1
            yc = y - crop_y1
            scale_x = new_width / (crop_x2-crop_x1)
            scale_y = new_height / (crop_y2-crop_y1)
            xr = int(xc * scale_x)
            yr = int(yc * scale_y)
            frame_ldmks[yr,xr] = 1
        frames_ldmks.append(frame_ldmks)
        frames.append(resized_frame)
    return frames_ldmks, frames

def get_landmarks(video_path, video_length=8, resolution=(512,512)):
    json_path = video_path.replace('gifs', 'landmarks').replace('.gif','.json')
    with open(json_path) as f:
        info = json.load(f)
    reader = imageio.mimread(video_path)
    
    frames_ldmks = []
    frames = []
    for f_idx in range(video_length):
        frame = reader[f_idx]
        width, height =  frame.shape[:2]
        new_width, new_height = resolution
        
        ldmks = info['frames'][f_idx]['faces'][-1]['landmarks']

        if width != height:
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

            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
        else:
            cropped_frame = frame
            crop_x1 = 0
            crop_y1 = 0
            crop_x2 = width
            crop_y2 = height
        print(cropped_frame.shape, (new_height, new_width))
        resized_frame = cv2.resize(cropped_frame, (new_height, new_width))
        
        frame_ldmks = np.zeros((new_height, new_width,1))
        for x,y in ldmks:
            xc = x - crop_x1
            yc = y - crop_y1
            scale_x = new_width / (crop_x2-crop_x1)
            scale_y = new_height / (crop_y2-crop_y1)
            xr = int(xc * scale_x)
            yr = int(yc * scale_y)
            frame_ldmks[yr,xr] = 1
        frames_ldmks.append(frame_ldmks)
        frames.append(resized_frame)
    return frames_ldmks, frames

video_path = '/dataset00/Videos/smile/gifs/smile_23.gif'
video_path = '/work00/AnimateDiff-adapter/zy_datas/zy_wink_new.gif'
video_path = '/dataset00/Videos/smile/gifs/smile_375.gif'
frames_ldmks, frames = get_landmarks(video_path)
imageio.mimsave('smile_375_show.gif', frames, fps=5)
# for i in range(len(frames_ldmks)):
#     print(frames[i].max(), frames[i].min())
#     cv2.imwrite(f"{i}_img.png", (frames[i]*(1-frames_ldmks[i])).astype('uint8')[:,:,:3])
#     cv2.imwrite(f"{i}_ldmks.png", (255*frames_ldmks[i]).astype('uint8'))      