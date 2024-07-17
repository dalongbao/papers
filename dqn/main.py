import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

import os
import sys
import pickle
import time
import random
import numpy as np
from collections import deque, namedtuple

import gymnasium as gym
from itertools import count

from dqn import DQN, ReplayBuffer

env = gym.make('CartPole-v1')
n_actions = env.action_space.n # output_dims

state, info = env.reset()
n_observations = len(state) # input_dims

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

"""Hyperparameters"""
lr = 1e-4
epoch = 100
batch_size = 128
discount = 0.9
# include epsilon decay later - start, end, decay

"""Rest of the model"""
model = DQN(n_observations, n_actions)
mx.eval(model.parameters())

optimizer = optim.AdamW(learning_rate=lr)
memory = ReplayBuffer(10000)

def loss_fn(X, y):
    return mx.mean(nn.losses.mse_loss(X, y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def optimize():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = mx.concatenate(batch.state)
    action_batch = mx.concatenate(batch.action)
    reward_batch = mx.concatenate(batch.reward)

    non_final_mask = mx.array(tuple(map(lambda s: s is not None for s in batch.next_state)), dtype=bool_)
    non_final_next_states = mx.concatenate([s for s in batch.next_state if s is not None])

    # Computing the predictions
    next_state_values = mx.zeros(batch_size)
    next_state_values[non_final_mask] = mx.argmax(model(non_final_next_states), axis=1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss, grad = loss_and_grad_fn(model, next_state_values, expected_state_action_values)  
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state)

    print('Loss: ', loss.item())

"""
for final states return the immediate reward
for non-terminal states return the reward + discounted q value of next state

q value is given by bellman, of course
"""

for e in range(epoch):
    state, info = env.reset()
    state = mx.expand_dims(mx.array(state, dtype=mx.float32), axis=0)

    for c in count():
        action = model(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = mx.array([reward])
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = mx.expand_dims(mx.array([observation], dtype=mx.float32), axis=0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize()

print('Complete')
