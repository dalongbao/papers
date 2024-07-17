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
        if x.ndim == 1:
            x = x.reshape(1, -1)
        c, h = prev_state
        x = mx.concatenate((h, x), axis=1) # first axis is usually batch size

        g1 = self.sigmoid(self.W1(x))
        g2 = self.sigmoid(self.W2(x))
        g3 = self.tanh(self.W3(x))
        g4 = self.sigmoid(self.W4(x))

        c_cur = mx.add(mx.multiply(c, g1), mx.multiply(g2, g3)) # this gets passed on
        h_cur = mx.multiply(g4, self.tanh(c_cur)) # this also gets passed on

        return c_cur, h_cur 

class LSTMLayer(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.cell = LSTM(input_dims, hidden_dims)

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

class LSTMRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_dims: int, num_layers: int): # add dropout later
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
    
        self.embedding = nn.Embedding(vocab_size, hidden_dims)
        self.layers = [LSTM(hidden_dims, hidden_dims) for _ in range(num_layers)]
        self.fc = nn.Linear(hidden_dims, vocab_size)
        
    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_size, seq_len = x.shape # batch_size, seq_len, input_size
        x = self.embedding(x)
        h = [mx.zeros((batch_size, self.hidden_dims)) for _ in range(self.num_layers)]
        c = [mx.zeros((batch_size, self.hidden_dims)) for _ in range(self.num_layers)]

        outputs = []

        for i in range(seq_len):
            input_i = x[:, i, :]
            for layer in range(self.num_layers):
                c[layer], h[layer] = self.layers[layer](input_i, (c[layer], h[layer]))
                input_i = h[layer]
            outputs.append(h[-1])
        x = self.fc(mx.array(outputs)) #self.fc(mx.stack(outputs, axis=1)[:, -1])
        return x
