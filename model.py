import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT(nn.Module):
    # GPT2 implementation
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.wpe = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config['num_layers'])])
        self.ln = nn.LayerNorm(config['hidden_size'])
        self.fc = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.apply(self._init_weights)
        # weight-tying
        self.wte.weight = self.fc.weight
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def forward(self, x, y=None):
        device = x.device
        b_sz, t_sz = x.shape
        pos = torch.arange(0, t_sz, device=x.device) # T
        token_emb = self.wte(x) # B, T, H
        pos_emb = self.wpe(pos) # T, H
        x = self.drop(token_emb + pos_emb) # B, T, H
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if y is not None:
            logits = self.fc(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_y = y[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
            # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
            loss = loss.mean()
        else:
            # (B, T, H) -> (B, 1, V)
            logits = self.fc(x[:, [-1], :])
            loss = None
        return logits, loss

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
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

class DecoderBlock(nn.Module):
    # Decoder block consisting of attention and mlp sub-blocks 
    # Decoder block interweaves sub-blocks with residual paths and layer norms
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config['hidden_size'])
        self.attn = MultiheadAttention(config)
        self.ln2 = nn.LayerNorm(config['hidden_size'])
        self.ffn = MLP(config)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class MultiheadAttention(nn.Module):
    # Attention sub-block of decoder block
    def __init__(self, config):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["attn_dropout"],
        )
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        # x: B, T, H
        attn_output, _ = self.mha(x, x, x, attn_mask=attention_mask)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class MLP(nn.Module):
    # Feedforward NN sub-block of decoder block
    def __init__(self, config):
        super().__init__()
        embed_dim = config['hidden_size']
        intermediate_size = 4 * embed_dim

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, embed_dim),
            nn.Dropout(0.1)
        )
    def forward(self, hidden_states):
        return self.layers(hidden_states)