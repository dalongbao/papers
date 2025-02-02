import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import os
import math
import time
import numpy as np
from pathlib import Path

from utils import save_checkpoint, load_checkpoint, get_dataset
from model import ViT, ViTConfig

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

dataset = 'imagenet'
checkpoint_dir = Path('./ckpts')
checkpoint_dir.mkdir(exist_ok=True)

num_epochs = 2400
eval_iter = 20
save_iter = 20
batch_size = 64 # GPU mem / (4 * input tensor size * no. parameters) 
num_workers = 2
learning_rate = 3e-3 * (256/4096) 
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.3 
warmup_steps = 10000 * (256/4096) 
grad_clip = 1.0
gradient_accumulation_steps = 8
continue_training = False

image_size = (224, 224)
patch_size = (16, 16)
num_classes = 100
dim = 768
dim_head = 64
depth = 12
num_heads = 12
hidden_dim = 3072
channels = 3
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"using device: {device}")
print(f"using cuda version: {torch.version.cuda}" if torch.cuda.is_available() else "")

training_data, test_data = get_dataset(dataset)

# ViT-Base/16 (16 x 16 patches)
config = ViTConfig(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    dim_head=dim_head,
    depth=depth,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    channels=channels,
    pool='cls',
    dropout=dropout,
    emb_dropout=dropout
)

model = ViT(config)
model = model.to(device)
model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=betas,
    eps=eps,
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

start_epoch = 0
best_accuracy = 0.0
checkpoint_file = checkpoint_dir / "latest.pt"
best_model_file = checkpoint_dir / "best.pt"

if checkpoint_file.exists() and continue_training:
    print(f"Loading checkpoint from {checkpoint_file}")
    start_epoch, _, best_accuracy = load_checkpoint(model, optimizer, scheduler, "latest.pt")
    print(f"Resuming from epoch {start_epoch} with best accuracy {best_accuracy:.2f}")

t0 = time.time()
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    model.train()
    size = len(training_data)

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y) / gradient_accumulation_steps
        loss.backward()

        if (batch + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    if epoch % eval_iter == 0:
        model.eval()
        with torch.no_grad():
            size = len(test_data)
            num_batches = len(test_dataloader)
            test_loss, acc = 0.0, 0.0

            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                test_loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
        test_loss /= num_batches
        acc /= size
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"Epoch: {epoch} | Accuracy: {acc:.3f} | Avg loss: {test_loss:.3f} | Time taken: {dt*1000:.2f}ms") 
       

        if epoch % save_iter == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, test_loss, acc,
                "latest.pt"
            )
            
            if acc > best_accuracy:
                best_accuracy = acc
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch + 1, test_loss, acc,
                    "best.pt"
                )
                print(f"New best model saved with accuracy: {acc:.3f}")

print("Training completed!")
print(f"Best accuracy: {best_accuracy:.2f}%")
