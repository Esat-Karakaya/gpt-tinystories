from config8M import cfg
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from utils import collate_fn

batch_size = cfg.batch_size

# Load dataset
DS_NAME = "esat-krky/TinyStories_Turkish"
ds_pth = "./datasets/" + DS_NAME.rsplit("/", 1)[1]
try:
    train_dataset = load_from_disk(ds_pth + "/ts_train")
    print("loaded dataset from disk 💽")
except:
    print("heading to 🤗 to load dataset")
    train_dataset = load_dataset(DS_NAME)["train"]

tokenizer = cfg.tokenizer
max_context = cfg.model.max_position_embeddings

torch.manual_seed(cfg.seed)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, max_context),
    drop_last=True,
)