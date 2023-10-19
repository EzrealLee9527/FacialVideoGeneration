# replace_code

1. 需要先搭建 DECA 代码框架（https://github.com/yfeng95/DECA），并将 replace_code.py 文件放在 DECA/demos 文件夹下面
2. 从 tos 上下载模板视频与模板 code，例如： oss cp s3://ljj-sh/Datasets/Videos/videos_1600_gen/worker0/conditions/10040716   YOUR_PATH/10040716
3. 运行脚本： python demos/replace_code.py --savefolder YOUR_REFER_PATH --lora_path YOUR_LORA_PATH--rasterizer_type=pytorch3d --saveVis=True --saveDepth=True --saveImages=True 