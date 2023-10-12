import requests
from PIL import Image
from io import BytesIO

from controlnet_aux.processor import Processor

# load image
# url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

# response = requests.get(url)
# img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
img_path = 'pose.png'
img = Image.open(img_path).convert("RGB").resize((512, 512))

# load processor from processor_id
# options are:
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe", "dwpose"]
processor_id = 'scribble_hed'
processor = Processor(processor_id)

processed_image = processor(img, to_pil=True)
print('processed_image')
print(processed_image.save('scribble_hed_pose.png'))
print(processed_image)