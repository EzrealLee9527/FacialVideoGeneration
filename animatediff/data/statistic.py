
from megfile import smart_open, smart_exists
import msgpack
import os
from glob import glob

def get_file_info(file_path):
    meta_files = glob(
        os.path.join(file_path, "*.msgpack"))
    print(f"Msg pack file : {len(meta_files)}")
    meta_info_list = []
    for meta_f in meta_files:
        meta_info_list += load_msgpack_list(meta_f)
    return meta_info_list

def load_msgpack_list(file_path: str):
    loaded_data = []
    with smart_open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f,strict_map_key = False)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data
    
file_path = 's3://ljj-sh/Datasets/Videos/msgpacks/videos4000_gen_labels/videos4000_gen-worker7-99.msgpack'
all_datas = load_msgpack_list(file_path)
print(all_datas)
import pdb;pdb.set_trace()
print(all_datas)