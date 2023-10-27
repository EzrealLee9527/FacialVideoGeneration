# search_dir=/data00/bzd/DECA_llz/lora_datasets
# i=0;
# for entry in "$search_dir"/*
# do 
#   echo CUDA_VISIBLE_DEVICES=$((i++%4)) python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e $entry --savename $(basename $entry) &
# done
# wait

CUDA_VISIBLE_DEVICES=0 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/angelababy --savename angelababy &
CUDA_VISIBLE_DEVICES=1 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/angelina --savename angelina &
CUDA_VISIBLE_DEVICES=2 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/blackwidow --savename blackwidow &
CUDA_VISIBLE_DEVICES=3 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/elon --savename elon & 
wait
CUDA_VISIBLE_DEVICES=0 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/emlia_clark --savename emlia_clark &
CUDA_VISIBLE_DEVICES=1 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/emma --savename emma &
CUDA_VISIBLE_DEVICES=2 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/emma_stone --savename emma_stone &
CUDA_VISIBLE_DEVICES=3 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/fanbingbing --savename fanbingbing &
wait
CUDA_VISIBLE_DEVICES=0 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/hathaway --savename hathaway &
CUDA_VISIBLE_DEVICES=1 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/inch --savename inch &
CUDA_VISIBLE_DEVICES=2 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/jay --savename jay &
CUDA_VISIBLE_DEVICES=3 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/johnny_depp --savename johnny_depp &
wait

CUDA_VISIBLE_DEVICES=0 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/leonardo --savename leonardo &
CUDA_VISIBLE_DEVICES=1 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/liuhaocun --savename liuhaocun &
CUDA_VISIBLE_DEVICES=2 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/liushishi --savename liushishi &
CUDA_VISIBLE_DEVICES=3 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/liuyifei --savename liuyifei &
wait
CUDA_VISIBLE_DEVICES=0 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/mj38 --savename mj38 &
CUDA_VISIBLE_DEVICES=1 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/tom_holland --savename tom_holland &
CUDA_VISIBLE_DEVICES=2 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/yangmi --savename yangmi &
CUDA_VISIBLE_DEVICES=3 python3 replace_code_v2.py --savefolder ./stickers_templates_result/ -e /data00/bzd/DECA_llz/lora_datasets/zhaoliying --savename zhaoliying &
wait