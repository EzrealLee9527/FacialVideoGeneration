from PIL import Image, ImageSequence
import imageio
import numpy as np

def concatenate_gif(gif1,gif2,gif3, output_file):
    frames = []
    reader1 = imageio.mimread(gif1)
    reader2 = imageio.mimread(gif2)
    reader3 = imageio.mimread(gif3)
    video_length = len(reader1)
    for f_idx in range(video_length):
        f1 = reader1[f_idx]
        f2 = reader2[f_idx]
        f3 = reader3[f_idx]
        f = np.hstack((f1,f2,f3))
        frames.append(f)
    imageio.mimsave(output_file, frames, fps=5)



gif1 = '/work00/AnimateDiff-adapter/smile_375_show.gif'
gif2 = '/work00/AnimateDiff-adapter/samples/test_landmark_adapter-2023-09-01T03-54-37/sample/34-1girl-is-smiling,-upper-body,-beautiful-face,-straight-hair,-long.gif'
gif3 = '/work00/AnimateDiff-adapter/samples/test_landmark_adapter-2023-09-01T03-54-37/sample/32-1girl-is-smiling,-upper-body,-beautiful-face,-straight-hair,-long.gif'

gif_files = [gif1, gif2, gif3]
output_file = "landmarks_adapter_results.gif"
concatenate_gif(gif1, gif2, gif3, output_file)
