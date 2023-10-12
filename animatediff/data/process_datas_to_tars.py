import os
import random
import datetime
from multiprocessing import Process
from torchvision import datasets
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import ImageFolder
from webdataset import TarWriter


def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    stream = TarWriter(fname, **kwargs)
    size = 0
    for item in samples:
        size += stream.write(map_func(item))
    stream.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")
    return size


if __name__ == "__main__":
    root = "/gdata/ImageNet2012/train"
    items = []

    dataset = ImageFolder(root=root,  loader=lambda x:x)
    for i in range(len(dataset)):
        items.append(dataset[i])
    print(dataset[0],os.path.splitext(os.path.basename(dataset[0][0]))[0])

    def map_func(item):
        name, class_idx = item
        with open(os.path.join(name), "rb") as stream:
            image = stream.read()
        sample = {
            "__key__": os.path.splitext(os.path.basename(name))[0],
            "jpg": image,
            "cls": str(class_idx).encode("ascii")
        }
        return sample

    make_wds_shards(
        pattern="/userhome/tars/imagenet-1k-%06d.tar",
        num_shards=256, # 设置分片数量
        num_workers=8, # 设置创建wds数据集的进程数
        samples=items,
        map_func=map_func,
    )
