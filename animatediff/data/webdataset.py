import webdataset as wds
import torch
from tqdm import tqdm 

url = "pipe: aws --endpoint-url=http://oss.i.basemind.com s3 cp s3://ljj/c3m_train/00000.tar -"
#url = "pipe: aws --endpoint-url=http://oss.i.basemind.com s3 cp s3://collect-22040715001-data/laion400m_part0/05502.tar -"
dataset = wds.WebDataset(url).shuffle(1000).decode("rgb").to_tuple("jpg", "json", "txt")
print(isinstance(dataset, torch.utils.data.IterableDataset))

for _, json, txt in tqdm(dataset):
    #print(image.shape)
    print(txt)
    print(json)