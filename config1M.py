from transformers import GPTNeoConfig, AutoTokenizer, models

# Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration
modelcfg = GPTNeoConfig(
    vocab_size=50257,
    max_position_embeddings=300,
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

from dataclasses import dataclass

@dataclass
class TrainConfig:
    name: str
    model: models.gpt_neo.configuration_gpt_neo.GPTNeoConfig
    tokenizer: models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
    epoch: int
    batch_size: int
    learning_rate: float
    val_freq: int
    sample_size: int
    seed: int
    model_location: str


cfg = TrainConfig(
    name="1M",
    model=modelcfg,
    tokenizer=tokenizer,
    epoch=2,
    batch_size=32,
    learning_rate=1e-3,
    val_freq=400,
    sample_size=30,
    seed=64,
    model_location="models/sml/1M"
)