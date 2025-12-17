from transformers import GPTNeoConfig, AutoTokenizer, models
from utils import is_notebook

# Loading tokenizer
try:
    tokenizer_pth = "./models/local_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth)
except Exception as e:
    print("Error:", e)
    model_name = 'roneneldan/TinyStories'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if is_notebook()==False:
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
    save_freq: int
    sample_size: int
    seed: int
    model_location: str

modelcfg = GPTNeoConfig(
    vocab_size=50257,
    max_position_embeddings=300,
    hidden_size=256,
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

cfg = TrainConfig(
    name="8M",
    model=modelcfg,
    tokenizer=tokenizer,
    epoch=1,
    batch_size=32,
    learning_rate=1e-3,
    val_freq=200,
    save_freq=6000,
    sample_size=30,
    seed=64, # Needs to be the same pre&post checkpoint
    model_location="models/sml/8M"
)

if(cfg.sample_size>cfg.val_freq):
    assert "Bad config: sample_size may not be bigger than val_freq"