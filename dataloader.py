from configs.config import cfg
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from utils import collate_fn

# Load dataset
try:
    ds_pth = (
        "./datasets/" +
        cfg.ds_name.rsplit("/", 1)[1] +
        "/ts_train"
    )

    train_dataset = load_from_disk(ds_pth)
    print("loaded dataset from disk 💽")
except:
    print("loading dataset from 🤗")
    train_dataset = load_dataset(cfg.ds_name)["train"]

tokenizer = cfg.tokenizer
max_context = cfg.model.max_position_embeddings

torch.manual_seed(cfg.seed)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, max_context),
    drop_last=True,
)