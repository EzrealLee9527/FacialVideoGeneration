# FacialVideoGeneration

## Set up

animatediff 环境 + ```pip install pyfacer, megfile```
## 训练方式

1. 训练 adapter，首先更改 config 文件 ```configs/training/train_adapter_deca.yaml``` 中 ```s3_output_dir```、```validation_data.vis_img_path``` 等变量。
运行命令：
+ ddp 版本 ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29518 train_adapter_deca.py --config configs/training/train_adapter_deca.yaml```
+ noddp 版本 ```python train_adapter_deca_noddp.py --config configs/training/train_adapter_deca.yaml```

2. 训练 motion_module, 首先更改 config 文件 ```configs/training/train_motion_deca.yaml``` 中 ```s3_output_dir```、```validation_data.val_data_prefix```、```adapter_model.adapter_model_path``` 等变量。
运行命令:
+ ddp 版本 ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=5 --master_port 29517 train_motion_deca.py --config configs/training/train_motion_deca.yaml```

## inference 方式:

1. 替换 3D MM code, 详见 replace_code/README.md

2. 在 ```configs/prompts/mj_girl_id38_good_deca.yaml``` 中填写合适的模板，运行命令：
```python -m scripts.animate_use_adapter64_deca --config configs/prompts/mj_girl_id38_good_deca.yaml```
