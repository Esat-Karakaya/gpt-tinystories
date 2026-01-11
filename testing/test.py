from model import CausalLM
from configs.config import cfg
import torch
from dataloader import val_dataset
from torch.utils.data import DataLoader
from utils import collate_fn
import math

device="cuda"
sml = CausalLM(cfg).to(device)
file_name = cfg.model_location + "December 18 00_27"


tokenizer = cfg.tokenizer
max_context = cfg.model.max_position_embeddings

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer, max_context),
    drop_last=True,
)

sml.load_state_dict(torch.load(file_name, map_location=device))

test_loss = sml.calc_loader_loss(val_loader, 400, device)

print(test_loss)
print(math.exp(test_loss))