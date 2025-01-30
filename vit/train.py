import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
from pathlib import Path
import math
import numpy as np

from model import ViT, ViTConfig

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'config': model.config  
    }
    torch.save(checkpoint, checkpoint_dir / filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(checkpoint_dir / filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']

checkpoint_dir = Path('./ckpts')
checkpoint_dir.mkdir(exist_ok=True)

training_data = datasets.CIFAR10(
    root="../data/cifar10",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="../data/cifar10",
    train=False,
    download=True,
    transform=ToTensor()
)

num_epochs = 10
eval_iter = 1
batch_size = 32
learning_rate = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.01
best_accuracy = 0.0

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"using device: {device}")
print(torch.version.cuda)

config = ViTConfig(
    image_size=(32, 32),
    patch_size=(16, 16),
    num_classes=10,
    dim=768,
    dim_head=64,
    depth=8,
    num_heads=8,
    hidden_dim=3072,
    channels=3,
    pool='cls',
    dropout=0.,
    emb_dropout=0.
)

model = ViT(config)
model = model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=betas,
    eps=eps,
    weight_decay=weight_decay
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

start_epoch = 0
best_accuracy = 0.0
checkpoint_file = checkpoint_dir / "latest.pt"
best_model_file = checkpoint_dir / "best.pt"

if checkpoint_file.exists():
    print(f"Loading checkpoint from {checkpoint_file}")
    start_epoch, _, best_accuracy = load_checkpoint(model, optimizer, "latest.pt")
    print(f"Resuming from epoch {start_epoch} with best accuracy {best_accuracy:.2f}%")

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    model.train()
    size = len(training_data)

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    if epoch % eval_iter == 0:
        model.eval()
        with torch.no_grad():
            size = len(test_data)
            num_batches = len(test_dataloader)
            test_loss, acc = 0.0

            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                test_loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
            test_loss /= num_batches
            acc /= size
            print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n") 
        
        save_checkpoint(
            model, 
            optimizer, 
            epoch + 1, 
            test_loss, 
            accuracy, 
            "latest.pt"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                test_loss,
                accuracy,
                "best.pt"
            )
            print(f"New best model saved with accuracy: {accuracy:.2f}%")

print("Training completed!")
print(f"Best accuracy: {best_accuracy:.2f}%")
