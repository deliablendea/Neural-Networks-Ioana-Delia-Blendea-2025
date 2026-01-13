import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import cv2

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

def preprocess_state_outline(state):
    if len(state.shape) == 3 and state.shape[2] == 3:
        gray = state.sum(axis=2) / 3
    else:
        gray = state

    weights = torch.tensor([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
    weights = weights.reshape(1, 1, *weights.shape)
    gray = torch.tensor(gray).reshape(1, 1, *gray.shape)

    output = F.conv2d(gray.byte(), weights.byte())
    output = output.reshape(output.shape[2], output.shape[3])

    resized = cv2.resize(np.array(output), (84, 84))
    normalized = resized / 255.0

    return normalized

OUTLINE_COLOR = torch.tensor([84, 56, 71])

def preprocess_state_black_n_white(state):
    state = state[0:state.shape[0] - 110, 0:state.shape[1]]

    state[state == OUTLINE_COLOR] = 255
    state[state != 255] = 0

    state = cv2.resize(np.array(state), (84, 84))
    state = state[:, :, 1]
    normalized = state / 255.0

    return normalized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_network(network, target_network, replay_buffer, optimizer, batch_size, gamma, device):
    if replay_buffer.__len__() < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, final_state_check = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    final_state_check = torch.tensor(final_state_check, dtype=torch.float32).to(device)

    q_values = network(states).gather(1, actions).squeeze(-1)

    with torch.no_grad():
        max_next_q_values = target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - final_state_check)

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def stack_frames(frame, stacked_frames, new_epoch):
    if new_epoch:
        stacked_frames = np.stack([frame] * 4, axis=0)
    else:
        stacked_frames = np.concatenate((stacked_frames[1:, :, :], np.expand_dims(frame, 0)), axis=0)

    return stacked_frames

def train():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    wrapper = HumanRendering(env)

    STATE_SHAPE = (4, 84, 84)
    NR_ACTIONS = env.action_space.n
    GAMMA = 0.99
    LR = 0.0001
    BATCH_SIZE = 32

    OBSERVE = 1
    EXPLORE = 10000
    REPLAY_BUFFER = 50000

    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.0001
    SKIP_FRAMES = 2
    DISPLAY = False

    SAVE_INTERVAL = 20

    epochs = EXPLORE
    epsilon = INITIAL_EPSILON
    replay_buffer = deque(maxlen=REPLAY_BUFFER)

    q_network = QNetwork(STATE_SHAPE, NR_ACTIONS).to(device)
    target_q_network = QNetwork(STATE_SHAPE, NR_ACTIONS).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=LR)

    q_network.load_state_dict(torch.load("weights/best_weights.pth"))
    target_q_network.load_state_dict(torch.load("weights/best_weights.pth"))

    for epoch in range(420, epochs):
        env.reset()
        if DISPLAY and epoch > OBSERVE:
            wrapper = HumanRendering(env)
            wrapper.reset()
        state = preprocess_state_black_n_white(env.render())

        stacked_frames = stack_frames(state, None, True)

        epoch_reward = 0
        done = False
        pipes = 0

        while not done:
            if DISPLAY and epoch > OBSERVE:
                wrapper._render_frame()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(device)
                    action = q_network(state_tensor).argmax().item()

            _, reward, done, _, _ = env.step(action)
            pipes += reward == 1

            if action == 1:
                if not done:
                    for _ in range(SKIP_FRAMES):
                        _, frame_reward, frame_done, _, _ = env.step(0)
                        pipes += frame_reward == 1

                        if DISPLAY and epoch > OBSERVE:
                            wrapper._render_frame()

                        reward += frame_reward
                        done = frame_done

                        if done:
                            break

            next_state = preprocess_state_black_n_white(env.render())
            next_stacked_frames = stack_frames(next_state, stacked_frames, False)

            replay_buffer.append((stacked_frames, action, reward, next_stacked_frames, done))
            if replay_buffer.__len__() > REPLAY_BUFFER:
                replay_buffer.popleft()

            stacked_frames = next_stacked_frames
            epoch_reward += reward

            if epoch > OBSERVE:
                train_network(q_network, target_q_network, replay_buffer, optimizer, BATCH_SIZE, GAMMA, device)
                pass

            if epoch > FINAL_EPSILON and epoch > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                epsilon = max(FINAL_EPSILON, epsilon)

            done = done

        if epoch % SAVE_INTERVAL == 0:
            torch.save(q_network.state_dict(), f'weights/model_weights_{epoch}.pth')

        if epoch % SAVE_INTERVAL == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        print(f"Episode: {epoch} | Reward: {epoch_reward:.2f} | Pipes: {pipes}")

        if DISPLAY and epoch > OBSERVE:
            wrapper.close()

        env.close()

train()