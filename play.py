from model import CausalLM
from config1M import cfg
import torch

sml = CausalLM(cfg)
file_name = cfg.model_location + "December 14 09_38"
device="cuda"

sml.load_state_dict(torch.load(file_name, map_location=device))

print(sml)

exit()
tokenizer = cfg.tokenizer
idx = tokenizer.encode("Once upon a time, there was a little girl named Jane")
res = sml.generate(torch.tensor(idx), max_length=100, temperature=.8, top_k=12)
res_str = tokenizer.decode(res)

print(res_str)