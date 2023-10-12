train_celebv_remains_happy:
	CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29510 train.py --config configs/training/train_celebv_remains_happy.yaml

train_celebv_happy:
	CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29511 train.py --config configs/training/train_celebv_happy.yaml

train_celebv_remains_happy_ddp:
	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train.py --config configs/training/train_celebv_remains_happy.yaml

inference:
	CUDA_VISIBLE_DEVICES=2 python -m scripts.animate --config configs/prompts/9-leosamsMoonfile_filmGrain20-wink.yaml --W 512 --H 512 --L 8

# train_adapter:
# 	CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29511 train_adapter_ldmks.py --config configs/training/train_smile_face_512_adapter.yaml

# train_adapter_ddp:
# 	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train_adapter_ldmks.py --config configs/training/train_smile_face_512_adapter.yaml

train_motion_based_adapter_ddp_sota:
	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train_use_adapter.py --config configs/training/train_motion_based_adapter_smile_face_512.yaml

train_motion_based_adapter_sota:
	CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 train_use_adapter.py --config configs/training/train_motion_based_adapter_smile_face_512.yaml

eval_train_motion_based_adapter:
	CUDA_VISIBLE_DEVICES=1 python -m scripts.animate_use_adapter64 --config configs/prompts/mj_girl_id38_v2_1600.yaml --W 512 --H 512 --L 16

eval_train_motion_based_adapter_given_image:
	CUDA_VISIBLE_DEVICES=0 python -m scripts.animate_use_adapter_given_image --config configs/prompts/test_landmark_adapter_given_image.yaml --W 512 --H 512 --L 8 --strength 0.8


train_adapter_ddp:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr localhost --master_port 29513 train_adapter_ldmks.py --config configs/training/train_adapter_online_celebv_512_16_stride1_ddp.yaml

train_adapter:
	CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29511 train_adapter_ldmks.py --config configs/training/train_adapter_online_celebv_512_16_stride1.yaml
train_adapter0:
	CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29512 train_adapter_ldmks.py --config configs/training/train_adapter_online_celebv_512_16_stride1.yaml


train_motion_based_adapter_ddp:
	CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_port 29517 train_use_adapter.py --config configs/training/train_motion_based_adapter_emotionalgif_smile_512_8_stride1_ldmk_crop_finetune.yaml

train_motion_based_adapter:
	CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29518 train_use_adapter.py --config configs/training/train_motion_based_adapter_emotionalgif_smile_512_8_stride1_ldmk_crop_finetune.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_addr localhost --master_port 29514 train_adapter_ldmks.py --config configs/training/train_adapter_emotionalgif_512_8_stride1_ldmk_faceparsing_crop_ddp.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_adapter_ldmks.py --config configs/training/train_adapter_emotionalgif_smile_512_8_stride1_ldmk_crop_ddp.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29516 train_adapter_face_parsing.py --config configs/training/train_adapter_emotionalgif_smile_512_8_stride1_faceparsing_crop_ddp.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29517 train_use_adapter64.py --config configs/training/train_motion_based_adapter_smile_face_512_finetuneV2.yaml
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29516 train_use_adapter64_frames16.py --config configs/training/train_motion_based_adapter_smile_face_512_finetuneV2_16frames.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29517 train_use_adapter64_frames16.py --config configs/training/train_motion_based_adapter_1600_face_512_finetuneV2_16frames.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_addr localhost --master_port 29515 train_adapter_ldmks.py --config configs/training/train_adapter_emotionalgif_512_8_stride1_ldmk_faceparsing_crop_ddp.yaml

CUDA_VISIBLE_DEVICES=3,6, python -m torch.distributed.launch --nproc_per_node=2 --master_addr localhost --master_port 29515 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml

CUDA_VISIBLE_DEVICES=3, python -m torch.distributed.launch --nproc_per_node=1 --master_addr localhost --master_port 29516 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml
CUDA_VISIBLE_DEVICES=6, python -m torch.distributed.launch --nproc_per_node=1 --master_addr localhost --master_port 29517 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml
CUDA_VISIBLE_DEVICES=4, python -m torch.distributed.launch --nproc_per_node=1 --master_addr localhost --master_port 29518 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml

CUDA_VISIBLE_DEVICES=0, RANK=0, 
python train_adapter_ldmks_noddp.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml

CUDA_VISIBLE_DEVICES=4, python torch.distributed.launch --nproc_per_node=1 --master_addr localhost --master_port 29518 train_adapter_ldmks.py 
--config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml

python /root/miniconda3/lib/python3.10/site-packages/torch/distributed/launch.py --nproc_per_node=1 --master_addr localhost --master_port 29518 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29517 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml


CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --nproc_per_node=1 --master_port 29518 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml
CUDA_VISIBLE_DEVICES=4, python -m torch.distributed.launch --nproc_per_node=1 --master_port 29519 train_motion.py --config configs/training/train_motion_based_adapter_1600_face_512_finetuneV2_16frames_tos.yaml
CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --nproc_per_node=1 --master_port 29517 train_motion.py --config configs/training/train_motion_based_adapter_1600_face_512_finetuneV2_16frames_tos.yaml
CUDA_VISIBLE_DEVICES=1, python -m torch.distributed.launch --nproc_per_node=1 --master_port 29518 train_adapter_ldmks.py --config configs/training/train_adapter_new_dataset_512_16_stride1_ldmk.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python -m torch.distributed.launch --nproc_per_node=8 --master_port 29517 train_motion.py --config configs/training/train_motion_based_adapter_1600_face_512_finetuneV2_16frames_tos.yaml

python gen_conditions_v2.py --out_dir 's3://ljj-sh/Datasets/Frames/videos1600_gen' --input 's3://ljj-sh/Datasets/Videos/smile_video_1600.tar'
python gen_conditions_v2.py --out_dir 's3://ljj-test/Datasets/Frames/videos1600_gen' --input 's3://ljj-sh/Datasets/Videos/smile_video_1600.tar'

CUDA_VISIBLE_DEVICES=1 python -m scripts.animate_use_adapter64 --config configs/prompts/107_The_person_turns_sad_into_sad_ldmks_frames16_tos_lora1.0_black_hair.yaml --W 512 --H 512 --L 16
CUDA_VISIBLE_DEVICES=1 python -m scripts.animate_use_adapter64 --config configs/prompts/107_The_person_turns_sad_into_sad_ldmks_frames16_tos.yaml --W 512 --H 512 --L 16

CUDA_VISIBLE_DEVICES=0 python -m scripts.animate_use_adapter64 --config configs/prompts/107_The_person_turns_sad_into_sad_ldmks_frames16_tos_lora0.5.yaml --W 512 --H 512 --L 16
