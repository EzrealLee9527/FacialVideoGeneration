train_celebv_remains_happy:
	CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29510 train.py --config configs/training/train_celebv_remains_happy.yaml

train_celebv_happy:
	CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29511 train.py --config configs/training/train_celebv_happy.yaml

train_celebv_remains_happy_ddp:
	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train.py --config configs/training/train_celebv_remains_happy.yaml

inference:
	CUDA_VISIBLE_DEVICES=2 python -m scripts.animate --config configs/prompts/9-leosamsMoonfile_filmGrain20-wink.yaml --W 512 --H 512 --L 8

train_adapter:
	CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29511 train_adapter_ldmks.py --config configs/training/train_smile_face_512_adapter.yaml

train_adapter_ddp_sota:
	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train_adapter_ldmks.py --config configs/training/train_smile_face_512_adapter.yaml

train_motion_based_adapter_ddp_sota:
	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train_use_adapter.py --config configs/training/train_motion_based_adapter_smile_face_512.yaml

train_motion_based_adapter:
	CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 train_use_adapter.py --config configs/training/train_motion_based_adapter_smile_face_512.yaml

eval_train_motion_based_adapter:
	CUDA_VISIBLE_DEVICES=0 python -m scripts.animate_use_adapter --config configs/prompts/test_landmark_adapter.yaml --W 512 --H 512 --L 8

eval_train_motion_based_adapter_given_image:
	CUDA_VISIBLE_DEVICES=0 python -m scripts.animate_use_adapter_given_image --config configs/prompts/test_landmark_adapter_given_image.yaml --W 512 --H 512 --L 8 --strength 0.8


train_adapter_ddp:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr localhost --master_port 29513 train_adapter_ldmks.py --config configs/training/train_adapter_online_celebv_512_16_stride1_ddp.yaml

train_adapter:
	CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29511 train_adapter_ldmks.py --config configs/training/train_adapter_online_celebv_512_16_stride1.yaml
train_adapter0:
	CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr localhost --master_port 29512 train_adapter_ldmks.py --config configs/training/train_adapter_online_celebv_512_16_stride1.yaml
