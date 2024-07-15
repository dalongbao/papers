import mlx
import mlx.nn as nn
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

"""
Structure:
    * Image (28, 28, 1)
    * Conv: (28, 28, 6), kernel_size=5, padding=2
    * Sigmoid
    * Pool: (14, 14, 6), kernel_size=2, stride=2
    * Conv: (10, 10, 16), kernel_size=5, no padding
    * Sigmoid
    * Pool: (5, 5, 16), kernel_size=2, stride=2
    * Flatten
    * Dense: 120 FC neurons
    * Sigmoid
    * Dense: 84 FC neurons
    * Sigmoid
    * Dense: 10 FC neurons
    * Output (10, 1)
"""

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = mx.flatten()

        self.dense1 = nn.Linear(input_dims=400, output_dims=120)
        self.dense2 = nn.Linear(input_dims=120, output_dims=84)
        self.dense3 = nn.Linear(input_dims=84, output_dims=10)

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        x = self.dense3(x)
        
        return x

model = LeNet()

lr = 1e-1

