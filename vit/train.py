import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from tqdm import tqdm
import torch.amp as amp

import os
import math
import time
import numpy as np
from pathlib import Path

from utils import save_checkpoint, load_checkpoint, get_dataset
from model import ViT, ViTConfig, KellyAdamW

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

dataset = 'cifar100'
checkpoint_dir = Path('./ckpts')
checkpoint_dir.mkdir(exist_ok=True)
print(f"using dataset {dataset}")

num_epochs = 4800
eval_iter = 20
save_iter = 20
batch_size = 32 # GPU mem / (4 * input tensor size * no. parameters) 
num_workers = 2

training_data, test_data, num_classes = get_dataset(dataset)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

learning_rate = 3e-3 * (256/4096) 
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.3 
decay_lr = True
lr_decay_iters = len(train_dataloader) * num_epochs // 2 # revisit
min_lr = 6e-5 # revisit
warmup_iters = 10000 * (256/4096) 
grad_clip = 1.0
gradient_accumulation_steps = 4
continue_training = False

image_size = (224, 224)
patch_size = (16, 16)
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
optimizer = KellyAdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=betas,
    eps=eps,
    weight_decay=weight_decay
)

scaler = amp.GradScaler('cuda')

# poor man's warmup + scheduler, taken directly from nanoGPT
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

start_epoch = 0
best_accuracy = 0.0
checkpoint_file = checkpoint_dir / "latest.pt"
best_model_file = checkpoint_dir / "best.pt"

if checkpoint_file.exists() and continue_training:
    print(f"Loading checkpoint from {checkpoint_file}")
    start_epoch, _, best_accuracy = load_checkpoint(model, optimizer, "latest.pt")
    print(f"Resuming from epoch {start_epoch} with best accuracy {best_accuracy:.2f}")

t0 = time.time()
print("Starting training...")
for epoch in range(start_epoch, num_epochs):

    running_loss = 0.0
    lr = get_lr(epoch) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    size = len(training_data)
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        with torch.autocast(device_type='cuda'):
            pred = model(X)
            loss = criterion(pred, y) / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        running_loss += loss.item()

        if (batch + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # update optimizer with averaged running loss
            avg_running_loss = running_loss / ((batch + 1) / gradient_accumulation_steps)
            optimizer.running_loss.append(avg_running_loss)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # update optimizer with epoch loss
    epoch_loss = running_loss / (len(train_dataloader) / gradient_accumulation_steps)
    optimizer.epoch_losses.append(epoch_loss)

    progress_bar.set_postfix({
        'loss': running_loss * gradient_accumulation_steps / (batch + 1),
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % eval_iter == 0:
        model.eval()
        with torch.no_grad():
            size = len(test_data)
            num_batches = len(test_dataloader)
            test_loss, acc = 0.0, 0.0

            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
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
                model, optimizer, 
                epoch + 1, test_loss, acc,
                f"{dataset}_latest.pt"
            )
            
            if acc > best_accuracy:
                best_accuracy = acc
                save_checkpoint(
                    model, optimizer, 
                    epoch + 1, test_loss, acc,
                    f"{dataset}_best.pt"
                )
                print(f"New best model saved with accuracy: {acc:.3f}")

print("Training completed!")
print(f"Best accuracy: {best_accuracy:.2f}%")
