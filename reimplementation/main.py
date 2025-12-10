from transformers import GPTNeoConfig, GPTNeoModel

device = "cpu"

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


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

paramsize = sum(p.numel() for p in model.parameters())
paramsize -= model.transformer.wpe.weight.numel()
print(f"Parameter size: {paramsize/1e6:.1f}M")