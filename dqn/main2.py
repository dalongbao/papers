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

import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from itertools import count

from dqn import DQN, ReplayBuffer

"""Preprocessing"""
class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, frame_size=(84, 84)):
        super().__init__(env)
        self.frame_size = frame_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*frame_size, 1), dtype=np.uint8)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._process_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._process_observation(observation), reward, terminated, truncated, info

    def _process_observation(self, observation):
        grayscale = np.mean(observation, axis=2).astype(np.uint8)
        scaled = self._resize_and_crop(grayscale, self.frame_size)
        return np.expand_dims(scaled, axis=-1)

    @staticmethod
    def _resize_and_crop(image, size):
        h, w = image.shape
        new_h, new_w = size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return image[top:top+new_h, left:left+new_w]

env = gym.make('ALE/Breakout-v5')
env = AtariPreprocessing(env, frame_size=(84, 84))
env = FrameStack(env, num_stack=4)
n_actions = env.action_space.n

state, info = env.reset()
n_observations = len(state)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

"""Hyperparameters"""
lr = 1e-4
epochs = 100
batch_size = 128
discount = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

"""Model setup"""
model = DQN(n_observations, n_actions)
mx.eval(model.parameters())

optimizer = optim.AdamW(learning_rate=lr)
memory = ReplayBuffer(10000)

def loss_fn(predictions, targets):
    return mx.mean(nn.losses.mse_loss(predictions, targets))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def optimize():
    if memory.len() < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = mx.stack([mx.array(s) for s in batch.state])
    action_batch = mx.array(batch.action)
    reward_batch = mx.array(batch.reward)

    non_final_mask = mx.array([s is not None for s in batch.next_state])
    non_final_next_states = mx.stack([mx.array(s) for s in batch.next_state if s is not None])

    state_action_values = model(state_batch).squeeze()[mx.arange(batch_size), action_batch]

    next_state_values = mx.zeros(batch_size)
    if len(non_final_next_states) > 0:
        next_state_values[non_final_mask] = mx.max(model(non_final_next_states), axis=1)

    expected_state_action_values = (next_state_values * discount) + reward_batch

    loss, grad = loss_and_grad_fn(state_action_values, expected_state_action_values)  
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state)
    
    return loss.item()

def select_action(state, epsilon):
    if random.random() < epsilon:
        return mx.array([env.action_space.sample()])
    else:
        with mx.no_grad():
            return model(mx.expand_dims(state, 0)).argmax(axis=1)

"""Training loop"""
epsilon = epsilon_start

for epoch in range(epochs): 
    state, info = env.reset()
    state = mx.array(state).transpose(3, 1, 2, 0)

    episode_reward = 0
    losses = []

    for t in count():
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        done = terminated or truncated

        next_state = mx.array(next_state).transpose(3, 1, 2, 0) if not done else None

        memory.push(state, action.item(), next_state, reward)

        state = next_state

        loss = optimize()
        if loss is not None:
            losses.append(loss)

        if done:
            break

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    avg_loss = sum(losses) / len(losses) if losses else 0
    print(f'Episode: {epoch+1} | Reward: {episode_reward} | Average Loss: {avg_loss:.3f} | Epsilon: {epsilon:.3f}')

print('Training complete')

# Optional: Save the trained model
mx.save("breakout_dqn.npz", model.parameters())
