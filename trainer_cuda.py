from config8M import cfg
from model import CausalLM
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from utils import save_model, log_state
from dataloader import train_loader, val_loader

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("running on", device)

sml = CausalLM(cfg)
sml.to(device)

#grad scaling
scaler = torch.amp.GradScaler(device=device)

optim = torch.optim.Adam(sml.parameters(), lr=cfg.learning_rate)
batch_cnt=0
train_losses=[]
val_losses=[]
train_batches_loss=[0]*cfg.sample_size
for epoch in range(cfg.epoch):
    for x, y in tqdm(train_loader):

        optim.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = sml.calc_loss(x, y, device)

        # Note: TPU's don't need a scaler
        scaler.scale(loss).backward()
        scaler.step(optim)

        train_batches_loss[batch_cnt%cfg.sample_size] = loss.item()
        batch_cnt+=1
        
        if batch_cnt%cfg.val_freq==0:
            train_loss = log_state(
                model=sml, cfg=cfg,
                train_loss_cache=train_batches_loss,
                batch_cnt=batch_cnt
            )
            train_losses.append(train_loss)
        
        if batch_cnt%cfg.save_freq==0:
            save_model(
                model=sml, cfg=cfg, optim=optim,
                train_losses=train_losses,
                epoch=epoch, batch_cnt=batch_cnt
            )
        
        scaler.update()

sml.eval()
model_file = cfg.model_location+datetime.now().strftime("%B %d %H:%M")
torch.save(sml.state_dict(), model_file)

#==================
#    Plotting
#==================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()