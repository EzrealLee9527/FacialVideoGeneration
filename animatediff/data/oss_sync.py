from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob, smart_copy
import os


# src_path = 's3://ljj-sh/Datasets/Videos/videos1600_gen/worker0/99.msgpack'
# dst_path = 's3://ljj-sh/Datasets/Videos/videos_231002/msgpacks/videos1600_gen-worker0-99.msgpack'

src_dir = 's3://ljj-sh/Datasets/Videos/videos4000_gen/'
dst_dir = 's3://ljj-sh/Datasets/Videos/msgpacks/videos4000_gen_labels/'

src_worker_pathes = smart_glob(src_dir+'worker*')
print(src_worker_pathes)
for src_worker_path in src_worker_pathes:
    src_msg_pathes = smart_glob(os.path.join(src_worker_path, '*.msgpack'))
    for src_msg_path in src_msg_pathes:
        print('src: ', src_msg_path)
        dst_msg_path = os.path.join(dst_dir, ('-').join(src_msg_path.split('/')[-3:]))
        print('dst: ', dst_msg_path)
        smart_sync(src_msg_path, dst_msg_path)

# local_path = '/work00/FacialVideoGeneration/outputs/train_adapter_new_dataset_512_16_stride1_ldmk-2023-10-02T16-57-15/checkpoints/checkpoint-epoch0-steps2.ckpt'
# s3_path = 's3://ljj-sh/work/llz_work/FacialVideoGeneration/outputs/train_adapter_new_dataset_512_16_stride1_ldmk-2023-10-02T16-57-15/checkpoints/checkpoint-epoch0-steps2.ckpt'
# # smart_copy(local_path, s3_path)
# os.system(f'aws --endpoint-url=https://tos-s3-cn-shanghai.volces.com s3 cp {local_path} {s3_path}')
# smart_remove(local_path)
