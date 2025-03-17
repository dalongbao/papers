import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Tuple

import math
import numpy as np

"""
cifar10

layers are residual layers
For the encoding part of the PixelCNN, the model uses 3 Resnet blocks consisting of 5 residual layers, with 2 × 2 downsampling in between
1/2/2/1?
All residual layers use 192 feature maps and a dropout rate of 0.5
- 192 filters
first layer to up/sample from previous, last layer to downsample to next


224 x 224 -> conv2d 7x7 stride 2 padding 1 -> 32 x 32 -> downsample by 2 -> next/downsample again then next -> repeat until 8x8 (smallest actually 4x4) -> reverse

connections are just residuals so they just need to be same dim. 
"""
class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(out_dim)
        )

        self.res = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.res = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_dim)
            )

    def forward(self, x):
        return F.relu(self.res(x) + self.block(x))


class DownBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int], 
        stride: int,
        padding: int, 
        dropout: float = 0.5
    ):
        super().__init__()

        # layer
        self.conv1 = nn.Conv2d(in_channels=in_channels ,out_channels=out_channels, kernel_size=7, stride=2, padding=1) 
        self.res1 = ResBlock(in_dim, out_dim, stride=stride, dropout=dropout)
        self.res2 = ResBlock(in_dim, out_dim, stride=stride, dropout=dropout)
        self.conv2 = nn.Conv2d(in_channels=in_channels ,out_channels=out_channels, kernel_size=7, stride=2, padding=1) 

        # residual connection
        self.

    def forward(self, x, res_x):
        x1 = self.conv1(x)
        x1 = self.res1(x1)

        x2 = self.res2(x1)
        x2 = self.conv2(x2)


class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):


class PixelCNNPP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):


