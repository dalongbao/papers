import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

import os
import sys
import pickle
import time

import numpy as np
from lstm import LSTMRNN

import tiktoken


def batch_iterate(batch_size, X):
    if not isinstance(X, mx.array):
        X = mx.array(X)

    perm = mx.array(np.random.permutation(len(X[1:])))
    for s in range(0, len(X[1:]), batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], X[1:][ids]

def loss_fn(model, X, targets):
    logits = model(X)
    logits = mx.squeeze(logits, 1)
    loss = nn.losses.cross_entropy(logits, targets)
    return mx.mean(loss)
    
def eval_fn(model, X, enc):
    logits = model(X)
    logits = np.array(mx.argmax(mx.squeeze(logits, 1), axis=-1)).tolist()
    tokens = enc.decode(logits)
    print('Logits: ', logits)
    print('Test generation:', ''.join(tokens))

def main():
    with open('../../mlxGPT/datasets/shakespeare/input.txt', 'r') as f:
        dataset = f.read()

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(dataset)

    """Hyperparameters"""
    epoch = 100
    batch_size = 256
    num_layers = 1
    hidden_dims = 256
    vocab_size = enc.n_vocab
    lr = 1e-4
    seq = mx.array(enc.encode("according to all known laws of aviation "))

    model = LSTMRNN(vocab_size, hidden_dims, num_layers)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


    for e in range(epoch):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, tokens):
            loss, grad = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state)

    #    accuracy = eval_fn(model,  # cba to implement this rn
        toc = time.perf_counter()
        print(f'Epoch: {e} | Loss: {loss.item():.3f} | Time: {(toc - tic):.3f}')
        eval_fn(model, seq, enc)

if __name__ == "__main__":
    print('Starting training...')
    main()
