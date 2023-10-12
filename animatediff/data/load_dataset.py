import msgpack
import megfile
from megfile import smart_open, smart_exists
import pickle
import torch
from PIL import Image, ImageDraw
import cv2

def gen_landmark_control_input(img_tensor, landmarks):
    indices_y = [y for x, y in landmarks]
    indices_x = [x for x, y in landmarks]
    tensor_indices_y = torch.tensor(indices_y).long().unsqueeze(0)
    tensor_indices_x = torch.tensor(indices_x).long().unsqueeze(0)
    ones = torch.ones(size=(1, len(tensor_indices_y)))
    img_tensor.scatter_(0, tensor_indices_y, ones)
    img_tensor.scatter_(1, tensor_indices_x, ones)
    return img_tensor

def load_msgpack_list(file_path: str):
    loaded_data = []
    with smart_open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f,strict_map_key = False)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data

data = load_msgpack_list('./videos1600_gen/worker0/0.msgpack')
# import pdb;pdb.set_trace()
# for frame in data[0]['frames']:
#     faces = frame['faces']
#     print('faces', faces)

landmarks_path = data[0]['frames'][0]['landmarks']
ldmk = pickle.load(open(landmarks_path,'rb'))
img_path = data[0]['frames'][0]['img']
img = cv2.imread(img_path)
h,w,c = img.shape
landmarks = ldmk[0]['landmark_2d_106']
def gen_landmark_control_input(img_tensor, landmarks):
    cols = torch.tensor([int(y) for x,y in landmarks])
    rows = torch.tensor([int(x) for x,y in landmarks])
    img_tensor = img_tensor.index_put_(indices=(cols, rows), values=torch.ones(106))
    return img_tensor
gen_ldmk = gen_landmark_control_input(torch.zeros((1920,1080)), ldmk[0]['landmark_2d_106'])
cv2.imwrite('aa.png', (gen_ldmk.unsqueeze(-1)*255).numpy().astype('uint8'))
import pdb;pdb.set_trace()

ldmk[0]['landmark_2d_106']
print(data)
for i in range(data[0]['num_frames']):
    with open(data[0]['frames'][i]['faces'], "rb") as file:
        loaded_faces_data = pickle.load(file)
    for face_i in range(len(loaded_faces_data)):
        print(loaded_faces_data[face_i][f'face{face_i}']['confidence'])
