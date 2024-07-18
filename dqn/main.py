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
        # Convert to grayscale
        grayscale = np.mean(observation, axis=2).astype(np.uint8)
        
        # Resize and crop
        scaled = self._resize_and_crop(grayscale, self.frame_size)
        
        # Add channel dimension
        return np.expand_dims(scaled, axis=-1)

    @staticmethod
    def _resize_and_crop(image, size):
        # Assuming the image is larger than 84x84, we'll crop the center
        h, w = image.shape
        new_h, new_w = size
        
        # Calculate crop dimensions
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        # Crop the center of the image
        cropped = image[top:top+new_h, left:left+new_w]
        
        return cropped

env = gym.make('ALE/Breakout-v5') # make grayscale later
env = AtariPreprocessing(env, frame_size=(84, 84))
env = FrameStack(env, num_stack=4)
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
    if memory.len() < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = mx.array(batch.state).squeeze(1)
    action_batch = mx.array(batch.action)
    reward_batch = mx.array(batch.reward)

    non_final_mask = mx.array([s is not None for s in batch.next_state], dtype=mx.bool_)
    non_final_next_states = mx.array([s for s in batch.next_state if s is not None]).squeeze(1)

    state_action_values = model(state_batch).squeeze()[mx.arange(batch_size), action_batch]

    # Computing the predictions
    next_state_values = mx.zeros(batch_size)
    print(non_final_next_states.shape)
    print(non_final_mask.shape)
    if len(non_final_next_states) > 0:
        next_state_values[non_final_mask] = mx.max(model(non_final_next_states), axis=1)

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss, grad = loss_and_grad_fn(state_action_values, expected_state_action_values)  
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()

"""
for final states return the immediate reward
for non-terminal states return the reward + discounted q value of next state

q value is given by bellman, of course

so right now it's being sampled in two places - the optimization and the generation

during the transitions it takes input of 1, 84, 84, 4; during optimize it takes 128, 84, 84, 4


"""

warmup_steps = batch_size

for e in range(epoch): 
    tic = time.perf_counter()
    state, info = env.reset()
    state = mx.array(np.array(state), dtype=mx.float32).transpose(3, 1, 2, 0)

    losses = []
    episode_reward = 0

    for c in count():
        action = model(state).argmax()
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = mx.array([reward])
        episode_reward += reward.item()
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = mx.array(np.array(observation), dtype=mx.float32).transpose(3, 1, 2, 0)

        memory.push(state.tolist(), action.item(), next_state.tolist() if next_state is not None else None, reward.item())

        state = next_state

        if c >= warmup_steps:
            loss = optimize()
            if loss is not None:
                losses.append(loss)
                counter += 1

        if done:
            break
    
    toc = time.perf_counter()
    print(f'Episode: {e} | Average Loss: {(np.mean(losses)):.3f} | Time: {(toc - tic):.3f}')

print('Complete')
