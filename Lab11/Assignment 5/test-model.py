import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import cv2
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms

import random
import numpy as np
from collections import deque

from qnetwork import QNetwork
from humanrendering import HumanRendering

OUTLINE_COLOR = torch.tensor([84, 56, 71])

def preprocess_state_black_n_white(state):
    state = state[0:state.shape[0] - 110, 0:state.shape[1]]

    state[state == OUTLINE_COLOR] = 255
    state[state != 255] = 0

    state = cv2.resize(np.array(state), (84, 84))
    state = state[:, :, 1]
    normalized = state / 255.0

    return normalized

def stack_frames(frame, stacked_frames, new_epoch):
    if new_epoch:
        stacked_frames = np.stack([frame] * 4, axis=0)
    else:
        stacked_frames = np.concatenate((stacked_frames[1:, :, :], np.expand_dims(frame, 0)), axis=0)

    return stacked_frames

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")

    STATE_SHAPE = (4, 84, 84)
    NR_ACTIONS = env.action_space.n

    q_network = QNetwork(STATE_SHAPE, NR_ACTIONS).to(device)
    q_network.load_state_dict(torch.load("weights/best_weights.pth"))
    q_network.eval()

    high_scores = []
    total_pipes_all_runs, total_reward_all_runs, epoch = 0, 0, 0

    while True:
        env.reset()
        wrapper = HumanRendering(env)
        wrapper.reset()

        state = preprocess_state_black_n_white(env.render())
        stacked_frames = stack_frames(state, None, True)

        done = False
        pipes = 0
        epoch_reward = 0

        while not done:
            wrapper._render_frame()

            with torch.no_grad():
                state_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(device)
                action = q_network(state_tensor).argmax().item()

            _, reward, done, _, _ = env.step(action)
            pipes += (reward == 1)
            epoch_reward += reward

            if action == 1 and not done:
                for _ in range(2):
                    _, f_reward, f_done, _, _ = env.step(0)
                    pipes += (f_reward == 1)
                    done = f_done
                    if done: break

            next_state = preprocess_state_black_n_white(env.render())
            stacked_frames = stack_frames(next_state, stacked_frames, False)

        total_pipes_all_runs += pipes
        total_reward_all_runs += epoch_reward
        avg_pipes = total_pipes_all_runs / (epoch + 1)
        avg_reward = total_reward_all_runs / (epoch + 1)

        high_scores.append(pipes)
        high_scores.sort(reverse=True)
        top_5 = high_scores[:5]

        print(f"Episode: {epoch}")
        print(f"Pipes: {pipes} | Average: {avg_pipes:.1f}")
        print(f"Reward: {epoch_reward:.1f} | Average: {avg_reward:.1f}")
        print(f"Top 5 pipes: {top_5}")
        print()

        epoch += 1
        wrapper.close()

test()