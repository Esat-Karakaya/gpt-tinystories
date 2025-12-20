from config8M import cfg
from model import CausalLM
import torch

print("begining execution")

device = "cuda"
tokenizer = cfg.tokenizer
model = CausalLM(cfg).to(device)
state_dict = torch.load("models/sml/8MDecember 19 13_32", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

idx = tokenizer.encode("Bir zamanlar Lily adında küçük bir kız varmış. Dışarıda çimlerde")

res = model.generate(torch.tensor(idx), 200, 1, 8, device)
print(tokenizer.decode(res.tolist()))