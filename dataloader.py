from config8M import cfg
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from utils import collate_fn, is_notebook

batch_size = cfg.batch_size

# Load dataset
DS_NAME = "umarigan/tinystories_tr"
ds_pth = "./datasets/" + DS_NAME.rsplit("/", 1)[1]
try:
    train_dataset = load_from_disk(ds_pth + "/ts_train")
except:
    print("loading dataset failed, heading to 🤗")
    train_dataset = load_dataset(DS_NAME)["train"]

    if is_notebook()==False:
        train_dataset.save_to_disk(ds_pth + "/ts_train")

split = train_dataset.train_test_split(
    test_size=0.001,
    seed=42
)
train_dataset = split["train"]
val_dataset = split["test"]


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
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, max_context),
    drop_last=True,
)