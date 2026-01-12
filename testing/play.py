import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config import cfg
from model import CausalLM
import torch

print("begining execution")

device = "cuda"
tokenizer = cfg.tokenizer
model = CausalLM(cfg).to(device)
state_dict = torch.load("/home/esat/Desktop/gpt-tinystories/models/sml/8MJanuary 11 21_31", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

idx = tokenizer.encode("Bir zamanlar Lily adında küçük bir kız varmış. Dışarıda çimlerde")

res = model.generate(torch.tensor(idx), 200, 1, 8, device)
#res = model.generate(torch.tensor(idx), 250, 0.9, int(tokenizer.vocab_size*.95), device)
print(tokenizer.decode(res.tolist()))