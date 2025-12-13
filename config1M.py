from transformers import GPTNeoConfig, AutoTokenizer

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

# Loading tokenizer
try:
    tokenizer_pth = "./models/local_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth)
except Exception as e:
    print("Error:", e)
    model_name = 'roneneldan/TinyStories'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("./models/local_tokenizer")

tokenizer.pad_token = tokenizer.eos_token
# causes: tokenizer.pad_token_id = tokenizer.eos_token_id

cfg = {
    "name":"1M",
    "model":modelcfg,
    "tokenizer": tokenizer,

    "batch_size": 64,
    "learning_rate": 1e-3
}
