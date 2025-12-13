from config1M import cfg
from model import CausalLM
import torch

device=torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

try:
    import torch_xla # type: ignore
    device = torch_xla.device()
except ImportError:
    pass

print("running on", device)