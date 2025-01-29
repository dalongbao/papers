import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import math
import numpy as np

from model import ViT

num_epochs = 10
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

model = ViT()
model = model.to(device)
model.eval()
optimizer = optim.AdamW(model.parameters())


