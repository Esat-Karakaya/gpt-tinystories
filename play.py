from model import CausalLM
from config8M import cfg
import torch

sml = CausalLM(cfg)
file_name = cfg.model_location + "December 18 00_27"
device="cuda"

sml.load_state_dict(torch.load(file_name, map_location=device))

tokenizer = cfg.tokenizer

idx = tokenizer.encode("Once upon a time, there was a boy named Timmy.")
res = sml.generate(torch.tensor(idx), max_length=60, temperature=1, top_k=5)
res_str = tokenizer.decode(res)

print(res_str)