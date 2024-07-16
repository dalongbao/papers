import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

import os
import sys
import pickle
import time

import numpy as np

"""
LSTM Block + MLP

LSTM:
    * take input x, output y
    * if there's no previous cells, no previous input is taken; otherwise, take the two outputs
    * 4 streams:
        1. sigmoid -> addition with prev -> addition with (3) -> (tanh -> multiplication with (4))/passed along
        2. sigmoid -> addition with (2)
        3. tanh -> addition operation -> addition with (1)
        4. sigmoid -> addition with (1)

Gates:
    * W1: forget
    * W2: Input
    * W3: cell
    * W4: output
"""

class LSTM(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.W1 = nn.Linear(input_dims + hidden_dims, hidden_dims)
        self.W2 = nn.Linear(input_dims + hidden_dims, hidden_dims)
        self.W3 = nn.Linear(input_dims + hidden_dims, hidden_dims)
        self.W4 = nn.Linear(input_dims + hidden_dims, hidden_dims)
    

    def __call__(self, x, prev_state):
        c, h = prev_state
        x = mx.concatenate((h, x), axis=1) # first axis is usually batch size

        g1 = self.sigmoid(self.W1(x))
        g2 = self.sigmoid(self.W2(x))
        g3 = self.tanh(self.W3(x))
        g4 = self.sigmoid(self.W4(x))

        c_cur = mx.add(mx.matmul(c, g1), mx.matmul(g2, g3)) # this gets passed on
        h_cur = mx.matmul(g4, self.tanh(c_cur)) # this also gets passed on

        return c_cur, h_cur 

class LSTMLayer(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.cell = LSTM(input_dims, hidden_dims)
        print(self.cell)

    def __call__(self, x):
        batch_size, seq_len = x.shape
        h = mx.zeros((batch_size, self.hidden_dims))
        c = mx.zeros((batch_size, self.hidden_dims))

        outputs = []

        for t in range(seq_len):
            c, h = self.cell(x[:, t, :], (c, h))
            outputs.append(h)

        return mx.stack(outputs, axis=1)

class MLP(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, hidden_dims: int, num_layers: int):
        super().__init__()
        layer_dims = [input_dims] + num_layers * [hidden_dims] + [output_dims]
        self.layers = [nn.Linear(idim, odim) for idim, odim in zip(layer_dims, layer_dims[1:])]
        print(self.layers)
        self.relu = nn.ReLU()

    # use the mx.compile thing later in main
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.relu(x)
        return x


