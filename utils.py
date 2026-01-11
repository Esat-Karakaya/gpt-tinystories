import torch
import math

def collate_fn(batch, tokenizer, max_context):
    stories = torch.utils.data.default_collate(batch)
    stories = stories["text"]
    
    input = tokenizer(
        stories,
        padding=True,
        truncation=True,
        max_length=max_context,
        return_tensors="pt"
    )["input_ids"]

    # Shift left
    target = input.clone()
    target[:, :-1] = input[:, 1:]
    target[:, -1] = tokenizer.pad_token_id

    return (
        input,
        target
    )

def log_state(cfg, train_loss_cache, batch_cnt):
    train_loss = sum(train_loss_cache)/cfg.sample_size

    print((
        f"step: {batch_cnt}, train_loss: {train_loss}, "
        f"perplexity: {math.exp(train_loss)}"
    ))
    return train_loss