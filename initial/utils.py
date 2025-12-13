import numpy as np
import random
import torch

def test_language_modeling(model, tokenizer, len=200, prompt=None, device='cuda'):
    if prompt is None:
        prompt = "One day, a little girl named Lily found a needle in her room."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    greedy_output = model.generate(input_ids, max_length=len)
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

def estimate_loss(model, tokenizer, valid_loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(40)
        for k,batch in enumerate(valid_loader):
            tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length = 256, truncation = True)['input_ids'].to(device)
            _, loss = model(tokenized,tokenized)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            losses[k] = loss.item()
            if k == 40 - 1 :
                break
    model.train()
    return losses.mean()

def save_checkpoint(model, optimizer, updates, filename="checkpoint.pt.tar"):
    state = {'updates': updates,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)

def load_checkpoint(model, filename, optim=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    updates = checkpoint['updates']
    return updates

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
