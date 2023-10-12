import cv2
from face_model import FaceAnalysis
import pickle

img = cv2.imread("/work00/AnimateDiff-adapter-mmv2/animatediff/data/out5/worker0/conditions/4607440/img_with_ldmks/204.png")
face_ana = FaceAnalysis()
face_ana.prepare(ctx_id=0, det_size=(640, 640))
res = face_ana.get(img, 1)
# plt.imshow(face_ana.draw_on(img, res)[...,::-1])
import numpy as np
# new_res = [{x:np.array(y) for x,y in res[0].items()}]
# new_res = {}
# for x,y in res[0].items():
#     new_res[x] = y
new_res = [dict(x) for x in res]
print(new_res)
print(res)

# data_faces = [
#     {
#         "bbox": np.array([470.61597, 649.89136, 1624.9832, 2333.2942], dtype=np.float32),
#         "kps": np.array(
#             [
#                 [1199.1661, 1287.3704],
#                 [1484.3256, 1309.6426],
#                 [1597.6842, 1545.4928],
#                 [1219.2703, 1868.5865],
#                 [1465.3011, 1895.136],
#             ],
#             dtype=np.float32,
#         ),
#         "det_score": 0.8264103,
#         "landmark_3d_68": np.array(
#             [
#                 [456.23383, 1362.99, -149.22049],
#                 [494.13434, 1542.246, -160.90511],
#                 [544.37671, 1714.9572, -182.27814],
#             ],
#             dtype=np.float32,
#         ),
#     }
# ]
with open('ldmk_f.pkl', 'wb') as file:
    pickle.dump(new_res, file)


import pdb;pdb.set_trace()

print('111')
import json
with open('ldmk_f.json', 'wb') as file:
    json.dump(res, file)