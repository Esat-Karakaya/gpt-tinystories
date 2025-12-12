from config1M import cfg
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

print("Launched!")

batch_size = cfg["batch_size"]

# Load dataset
try:
    train_dataset = load_from_disk("./datasets/ts_train")
    val_dataset = load_from_disk("./datasets/ts_val")
except:
    print("loading dataset failed, heading to 🤗")
    model_name = 'roneneldan/TinyStories'
    dataset = load_dataset(model_name)

    dataset["train"].save_to_disk("./datasets/ts_train")
    dataset["validation"].save_to_disk("./datasets/ts_val")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    

tokenizer = cfg["tokenizer"]
context_size = cfg["model"].max_position_embeddings

def collate_fn(batch, tokenizer):
    stories = torch.utils.data.default_collate(batch)
    stories = stories["text"]
    
    input = tokenizer(
        stories,
        padding=True,
        truncation=True,
        max_length=context_size,
        return_tensors="pt"
    )["input_ids"]

    # Shift left
    target = input.clone()
    target[:, :-1] = input[:, 1:]
    target[:, -1] = tokenizer.eos_token_id

    return (
        input,
        target
    )

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer),
    drop_last=True,
)