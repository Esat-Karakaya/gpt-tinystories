from transformers import GPTNeoConfig, AutoConfig

# Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration
modelcfg = GPTNeoConfig(
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

{
    "model":modelcfg,
    
    "bos_token_id": 50256,
    "embed_dropout": 0,
    "eos_token_id": 50256,
    "gradient_checkpointing": False,
    
    "torch_dtype": "float32",
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 50257,
    "window_size": 256,
    "batch_size": 32,
    "learning_rate": 1e-3
}
