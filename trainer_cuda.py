from config8M import cfg
from model import CausalLM
import torch
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

device=torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

try:
    import torch_xla # type: ignore
    device = torch_xla.device()
except ImportError:
    pass

print("running on", device)

epochs = cfg.epoch
val_freq = cfg.val_freq
sample_size = cfg.sample_size

saved_model_file = cfg.model_location+"placeholder"

sml = CausalLM(cfg)
sml.to(device)

# load model if possible
if(os.path.isfile(saved_model_file)):
    sml.load_state_dict(torch.load(saved_model_file, map_location=device))

from dataloader import train_loader, val_loader
optim = torch.optim.Adam(sml.parameters(), lr=cfg.learning_rate)

scaler = torch.amp.GradScaler("cuda")

batch_cnt=0
train_losses=[]
val_losses=[]
for epoch in range(epochs):
    for x, y in tqdm(train_loader):
        optim.zero_grad()

        # New: added scalers for saving vram
        with torch.amp.autocast("cuda"):
            loss = sml.calc_loss(x, y, device)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        batch_cnt+=1
        
        if batch_cnt%val_freq==0:
            with torch.amp.autocast("cuda"):
                train_loss = sml.calc_loader_loss(train_loader, sample_size, device)
                val_loss = sml.calc_loader_loss(val_loader, sample_size, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print((
                f"Epoch {epoch}, train_loss: {train_loss},"
                f"val_loss: {val_loss}, val_perplexity: {math.exp(val_loss)}"
            ))
        
        if batch_cnt%cfg.save_freq==0:
            sml.eval()
            model_file = cfg.model_location+datetime.now().strftime("%B %d %H:%M")
            torch.save(sml.state_dict(), model_file)
            sml.train()

sml.eval()
model_file = cfg.model_location+datetime.now().strftime("%B %d %H:%M")
torch.save(sml.state_dict(), model_file)

#==================
#    Plotting
#==================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()