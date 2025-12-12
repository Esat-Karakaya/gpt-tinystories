from config1M import cfg
from model import CausalLM
import torch

tokenizer = cfg["tokenizer"]

sml = CausalLM(cfg)
idx = tokenizer.encode("Hello")

generated = sml.generate(
    idx=torch.tensor(idx),
    max_length=15
)
print(generated.tolist())

print(
    tokenizer.decode(generated.tolist())
)