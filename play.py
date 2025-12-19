from config8M import cfg
from tqdm import tqdm

from datasets import load_dataset
from itertools import islice

step=0
tokenizer = cfg.tokenizer
idx_cnt = 0
max_step=200

print("loading dataset")
dataset = load_dataset(
    "umarigan/tinystories_tr",
    split="train",
    streaming=True
)

ds = list(islice(dataset, max_step))  # only 200 samples

print("loaded!")

for i,story in tqdm(enumerate(ds), total=max_step):
    str=story["text"]
    idx = tokenizer.encode(str)
    if i==0:
        print(idx)
    idx_cnt += len(idx)
    if max_step==step:
        break

print(idx_cnt)

# 105509