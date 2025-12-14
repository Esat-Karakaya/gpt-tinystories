from model import CausalLM
from config1M import cfg
import torch

sml = CausalLM(cfg)
file_name = cfg.model_location + "December 14 09_38"
device="cuda"

sml.load_state_dict(torch.load(file_name, map_location=device))

tokenizer = cfg.tokenizer
idx = tokenizer.encode()