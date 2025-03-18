import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataclasses import dataclass
#from einops import rearrange, repeat, einsum
from typing import Tuple

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
        kernel_size: Tuple[int, int] = (3, 3), 
        stride: int = 1,
        dropout: float = 0.5,
        final: bool =  False
    ):
        super().__init__()
        self.final = final

        # layer
        self.dropout = nn.Dropout(dropout)
        self.res1 = ResBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size=kernel_size, stride=stride)

        # convolution connections (where downsampling occurs). architectural design to keep all the convolution connections to the source
        assert out_channels % 2 == 0
        self.conv_channels = out_channels // 2
        self.res_conv = nn.Conv2d(out_channels, self.conv_channels, kernel_size=kernel_size, stride=2) # residual connection to next layer head
        self.out_conv = nn.Conv2d(out_channels, self.conv_channels, kernel_size=kernel_size, stride=2) # output downsampling

    def forward(self, x, res_x):
        x1 = self.dropout(self.res1(x))
        x2 = self.res2(x1)
        x2 = self.dropout(self.res3(x2) + res_x)
        res_out = self.res_conv(x1)
        out = self.out_conv(x2) 

        if self.final:
            return out, res_out, x2, x1

        return out, res_out

class UpBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int] = (3, 3), 
        stride: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()

        # layer
        self.dropout = nn.Dropout(dropout)
        self.res1 = ResBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size=kernel_size, stride=stride)

        # upsampling 
        assert out_channels % 2 == 0
        self.conv_channels = out_channels * 2

        self.res_conv = nn.ConvTranspose2d(out_channels, self.conv_channels, kernel_size=kernel_size, stride=2) # residual connection to next layer head
        self.out_conv = nn.ConvTranspose2d(out_channels, self.conv_channels, kernel_size=kernel_size, stride=2) # output upsampling 

    def forward(self, top, bottom, top_res, bottom_res):
        x1 = self.dropout(self.res1(top + top_res))
        x2 = self.res2(x1)
        x2 = self.dropout(self.res3(x2) + bottom + bottom_res)

        res_out = self.res_conv(x1)
        out = self.out_conv(x2) 

        return out, res_out


class PixelCNNPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(2, 3))
        self.conv2 = nn.Conv2d(3, 32, kernel_size=(1, 3))
        self.conv3 = nn.Conv2d(32, 224, kernel_size=(2, 3))
        self.conv4 = nn.Conv2d(32, 224, kernel_size=(1, 3))

        self.dblock1 = DownBlock(32, 32)
        self.dblock2 = DownBlock(16, 16)
        self.dblock3 = DownBlock(8, 8, final=True)

        self.ublock1 = UpBlock(8, 8)
        self.ublock2 = UpBlock(16, 16)
        self.ublock3 = UpBlock(32, 32)

    def forward(self, x):
        u1 = self.conv1(x)
        u2 = self.conv2(x)

        # outputs (bottom, top) so it must be swapped in 
        x1, x2 = self.dblock1(u2, u1)
        x3, x4 = self.dblock2(x2, x1)
        x5, x6, x7, x8 = self.dblock3(x4, x3) # out, res_out, skip2, skip1

        x7, x8 = self.ublock1(x6, x5, x8, x7)
        x9, x10 = self.ublock2(x8, x7, x4, x3)
        x11, x12 = self.ublock3(x10, x9, x2, x1)

        return self.conv3(x11) + self.conv4(x12) 
