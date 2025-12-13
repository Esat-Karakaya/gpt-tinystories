from config1M import cfg
import logging
from datetime import datetime

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model import GPT
from initial.utils import *  # contains all of the helper methods

device="cpu"
epochs = 3
seed = 3407

batch_size = cfg["batch_size"]
lr = cfg["learning_rate"]

# Set up logger
current_time = datetime.now().strftime("%m%d_%H%M%S")
log_filename = f"logs/training_{cfg["name"]}_{current_time}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load dataset and tokenizer
model_name = 'roneneldan/TinyStories'
dataset = load_dataset(model_name)
tokenizer = cfg["tokenizer"]

# Instantiate dataloader
train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)

# Instantiate model and optimizer
setup_seed(seed)
model = GPT(cfg)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

updates = 0
model_filename = f"models/model_{cfg["name"]}_{current_time}.pth"
resume_training = False
if resume_training:
    model_filename = ""
    logging.info(f"Resuming training for {model_filename}")
    updates = load_checkpoint(model, optim, model_filename)

logging.info(f"cfg_param: {cfg["name"]}, lr: {lr}, batch_size: {batch_size}, model_filename: {model_filename}, log_filename: {log_filename}, seed: {seed}, epochs: {epochs}")

# Training loop
for epoch in range(epochs):
    logging.info(f"Epoch: {epoch+1}")
    for batch in tqdm(train_loader):
        optim.zero_grad()
        tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=cfg["max_position_embeddings"], truncation=True)['input_ids'].to(device)
        _, loss = model(tokenized, tokenized)
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        loss.backward()
        optim.step()
        updates += 1
        if updates % 200 == 0:
            validation_loss = estimate_loss(model, tokenizer, valid_loader)
            tqdm.write(f"Train_{epoch+1}_{updates}: {validation_loss}")
            logging.info(f"Train_{epoch+1}_{updates}: {validation_loss}")
        if updates % 2000 == 0:
            save_checkpoint(model, optim, updates, model_filename)
    logging.info("TRAINING COMPLETE")
    logging.info("Computing final validation loss..")
    # Validation loop
    with torch.no_grad():
        loss_valid = 0
        for batch in tqdm(valid_loader):
            tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=512,truncation=True)['input_ids'].to(device)
            _, loss = model(tokenized, tokenized)
            loss_valid += loss.mean().item()
        logging.info(f"Final validation loss: {loss_valid / len(valid_loader)}")
        save_checkpoint(model, optim, updates, model_filename)

# Trained model output
test_language_modeling(model, tokenizer)