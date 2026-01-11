from transformers import GPTNeoConfig, AutoTokenizer, models, PreTrainedTokenizerFast

DS_NAME = 'roneneldan/TinyStories'
TOKENIZER_NAME = 'roneneldan/TinyStories'

# Loading tokenizer
tokenizer_pth = "./models/tokenizers/" + TOKENIZER_NAME.rsplit("/", 1)[1]

try:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_pth+"/tokenizer.json")
    print("loaded tokenizer from disk 💽")
except Exception as e:
    print("heading to 🤗 to load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

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
    ds_name: str

modelcfg = GPTNeoConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=300,
    hidden_size=256,
    num_layers=8,
    attention_types=[[["global", "local"], 4]],
    num_heads=16,
    intermediate_size=256,
    window_size=256,
    resid_dropout=0,
    embed_dropout=0,
    attention_dropout=0,
    classifier_dropout=0,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

cfg = TrainConfig(
    name="8M",
    model=modelcfg,
    tokenizer=tokenizer,
    epoch=1,
    batch_size=32,
    learning_rate=1e-3,
    val_freq=200,
    sample_size=30, # train loss is calculated by taking the last n train losses
    seed=64, # Needs to be the same pre&post checkpoint
    ds_name=DS_NAME,
    model_location="models/sml/8M"
)

if(cfg.sample_size>cfg.val_freq):
    assert "Bad config: sample_size may not be bigger than val_freq"