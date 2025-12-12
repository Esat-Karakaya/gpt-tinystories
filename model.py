from transformers import GPTNeoConfig, GPTNeoModel
import torch.nn as nn
import torch.nn.functional as F
import torch

# Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration
configuration = GPTNeoConfig(
    vocab_size=50257,
    max_position_embeddings=2048,
    hidden_size=64,
    num_layers=8,
    attention_types=[[["global", "local"], 4]],
    num_heads=16, # unknown
    intermediate_size=256,
    window_size=256,
    resid_dropout=0,
    embed_dropout=0,
    attention_dropout=0,
    classifier_dropout=0,
    bos_token_id=50256,
    eos_token_id=50256,
)

# Initializing a model (with random weights) from the EleutherAI/gpt-neo-1.3B style configuration
model = GPTNeoModel(configuration)

paramsize = sum(p.numel() for p in model.parameters())
paramsize -= model.wpe.weight.numel()
print(f"Parameter size: {paramsize/1e6:.1f}M")
print(configuration.hidden_size)

# tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

class CausalLM(nn.Module):
    def __init__(self, cfg: GPTNeoConfig):
        super().__init__()
        self.model = GPTNeoModel(cfg)
        self.out_head = nn.Linear( cfg.hidden_size, cfg.vocab_size, bias=False )

    def forward(self, x):
        out = self.model(x)
        out = self.out_head(out)
        return out
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.model.parameters())
        if non_embedding:
            n_params -= self.model.wpe.weight.numel()
        return n_params
    
    @torch.no_grad() 
    def generate(self, idx, max_length, temperature=1.0, top_k=None):
        # idx: B, T
        for _ in range(max_length):
            idx = idx[:, -self.config['window_size']:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, k = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx