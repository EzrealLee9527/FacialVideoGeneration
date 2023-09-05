
from self_collect_smile import SmileGif
local_path = '/data/users/liangjiajun/Datasets/'
mount_path = '/dataset00/'
meta_path = '/work00/AnimateDiff-adapter/animatediff/data/filelist.txt'
data_dir = mount_path + '/Videos/smile/gifs'
frame_stride = 1
dataset = SmileGif(meta_path, data_dir, frame_stride=frame_stride)
print('dataset size is ', len(dataset))
for i,x in enumerate(dataset)   :
    print(i)

