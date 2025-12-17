from config8M import cfg
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from utils import collate_fn, is_notebook

batch_size = cfg.batch_size

# Load dataset
try:
    train_dataset = load_from_disk("./datasets/ts_train")
    val_dataset = load_from_disk("./datasets/ts_val")
except:
    print("loading dataset failed, heading to 🤗")
    model_name = 'roneneldan/TinyStories'
    dataset = load_dataset(model_name)

    if is_notebook()==False:
        dataset["train"].save_to_disk("./datasets/ts_train")
        dataset["validation"].save_to_disk("./datasets/ts_val")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    

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
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, max_context),
    drop_last=True,
)