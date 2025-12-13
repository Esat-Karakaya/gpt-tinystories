import torch

@torch.no_grad()
def calc_loader_loss(model, loader, sample_size, device):
    model.eval()
    total_loss=0.0
    bnum=0
    for x, y in loader:
        loss = model.calc_loss(x, y, device)
        total_loss += loss.item()
        bnum+=1
        if bnum==sample_size: break
    
    model.train()
    return total_loss/bnum
