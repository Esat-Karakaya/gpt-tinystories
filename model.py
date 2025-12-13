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

class CausalLM(nn.Module):
    def __init__(self, cfg: GPTNeoConfig):
        super().__init__()
        modelcfg = cfg["model"]
        self.modelcfg=modelcfg
        self.eos_token_id = cfg["tokenizer"].eos_token_id
        self.pad_token_id = cfg["tokenizer"].eos_token_id
        self.model = GPTNeoModel(modelcfg)
        self.out_head = nn.Linear( modelcfg.hidden_size, modelcfg.vocab_size, bias=False )

    def forward(self, x):
        out = self.model(x)["last_hidden_state"]
        out = self.out_head(out)
        return out
    
    def calc_loss(self, x, y, dev):
        x, y = x.to(dev), y.to(dev)
        res = self.forward(x)
        # res: (B, context_size, dict_size)
        # y: (B, context_size)
        res = res.flatten(0, 1)
        y = y.flatten()
        loss = F.cross_entropy(res, y, ignore_index=self.pad_token_id)
        return loss
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.model.parameters())
        if non_embedding:
            n_params -= self.model.wpe.weight.numel()
        return n_params
    
    @torch.no_grad() 
    def generate(self, idx, max_length, temperature=1.0, top_k=None):

        # support for unbatched
        inp_dim = len(idx.shape)
        if inp_dim==1:
            idx = idx.unsqueeze(0)

        # idx: B, T
        for _ in range(max_length):
            idx = idx[:, -self.modelcfg.max_position_embeddings:]
            logits = self(idx)
            logits = logits[:, -1, :] / temperature

            # Filter logits with top_k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = v[:, [-1]]
                logits[logits < min_val] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)

            if idx[:, -1] == self.eos_token_id:
                break

        if inp_dim==1:
            idx = idx.squeeze()

        return idx