from config8M import cfg
from tqdm import tqdm

from datasets import load_dataset
from itertools import islice

step=0
tokenizer = cfg.tokenizer
idx_cnt = 0
max_step=1000

print("loading dataset")
dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

ds = list(islice(dataset, max_step))  # only 200 samples

print("loaded!")

for i,story in tqdm(enumerate(ds), total=max_step):
    str=story["text"]
    idx = tokenizer.encode(str)
    idx_cnt += len(idx)
    if max_step==step:
        break

print(idx_cnt)

# 39195
# 234420