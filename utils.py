import torch
import os
import math

# Source - https://stackoverflow.com/a
# Posted by Gustavo Bezerra, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-16, License - CC BY-SA 4.0
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


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

def save_model(model, cfg, optim, train_losses, val_losses, epoch, batch_cnt):
    model.eval()
    os.makedirs("models/sml", exist_ok=True)
    checkpoint_file = cfg.model_location+"checkpoint"
    torch.save({
        "model":model.state_dict(),
        "optim":optim.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch":epoch,
        "batch_cnt":batch_cnt,

    }, checkpoint_file)
    model.train()

def log_state(cfg, train_loss_cache, batch_cnt):
    train_loss = sum(train_loss_cache)/cfg.sample_size

    print((
        f"step: {batch_cnt}, train_loss: {train_loss}, "
        f"perplexity: {math.exp(train_loss)}"
    ))
    return train_loss