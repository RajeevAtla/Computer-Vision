import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pytorch_lightning as pl

import ale_py
import shimmy

# Hyperparameters
ENV_NAME = "ALE/MsPacman-v5"
LEARNING_RATE = 1e-4
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 500
MAX_STEPS = 10000

# Neural Network for the Q-function
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            conv_output = self.conv(dummy_input)
            self.conv_output_size = conv_output.numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv(x / 255.0)
        return self.fc(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Agent class
class DQNAgent:
    def __init__(self, state_shape, n_actions):
        self.n_actions = n_actions
        self.epsilon = EPSILON_START
        self.policy_net = DQN(state_shape, n_actions)
        self.target_net = DQN(state_shape, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.action_history = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state)
            action = torch.argmax(q_values).item()

        # Track action history for penalty evaluation
        self.action_history.append(action)
        if len(self.action_history) > 3:
            self.action_history.pop(0)

        return action

    def compute_penalty(self):
        if len(self.action_history) == 3 and len(set(self.action_history)) == 1:
            return -0.1  # Slight penalty for staying in place
        return 0.0

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Helper function to calculate Manhattan distance
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Lightning module
class DQNLightning(pl.LightningModule):
    def __init__(self, state_shape, n_actions):
        super(DQNLightning, self).__init__()
        self.pacman_agent = DQNAgent(state_shape, n_actions)
        self.ghost_agent = DQNAgent(state_shape, n_actions)

        self.env = gym.make(ENV_NAME, render_mode=None)
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.total_reward_pacman = 0
        self.total_reward_ghost = 0
        self.episode = 0

        self.automatic_optimization = False

    def forward(self, state):
        return self.pacman_agent.policy_net(state)

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()

        state, _ = self.env.reset()
        state = np.moveaxis(state, -1, 0)

        pacman_position = [0, 0]  # Placeholder for Pacman's position
        ghost_position = [0, 0]  # Placeholder for Ghost's position

        for step in range(MAX_STEPS):
            pacman_action = self.pacman_agent.select_action(state)
            next_state, reward, done, info, _ = self.env.step(pacman_action)

            pacman_penalty = self.pacman_agent.compute_penalty()
            pacman_reward = reward + pacman_penalty

            ghost_action = self.ghost_agent.select_action(state)

            pacman_position = info.get("pacman_position", pacman_position)
            ghost_position = info.get("ghost_position", ghost_position)

            distance = manhattan_distance(ghost_position, pacman_position)
            if distance < 3:
                ghost_reward = -0.5
            else:
                ghost_reward = 0.1

            if distance < 1:
                ghost_reward -= 1.0

            next_state = np.moveaxis(next_state, -1, 0)

            self.pacman_agent.replay_buffer.add(state, pacman_action, pacman_reward, next_state, done)
            self.ghost_agent.replay_buffer.add(state, ghost_action, ghost_reward, next_state, done)

            self.pacman_agent.train()
            self.ghost_agent.train()

            state = next_state
            self.total_reward_pacman += pacman_reward
            self.total_reward_ghost += ghost_reward

            if done:
                break

        self.pacman_agent.epsilon = max(EPSILON_MIN, self.pacman_agent.epsilon * EPSILON_DECAY)
        self.ghost_agent.epsilon = max(EPSILON_MIN, self.ghost_agent.epsilon * EPSILON_DECAY)

        if self.episode % TARGET_UPDATE == 0:
            self.pacman_agent.update_target_net()
            self.ghost_agent.update_target_net()

        self.episode += 1
        self.log("pacman_reward", self.total_reward_pacman)
        self.log("ghost_reward", self.total_reward_ghost)

    def configure_optimizers(self):
        return [self.pacman_agent.optimizer, self.ghost_agent.optimizer]

if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode=None)
    state_shape = env.observation_space.shape
    state_shape = (state_shape[2], state_shape[0], state_shape[1])
    n_actions = env.action_space.n

    model = DQNLightning(state_shape, n_actions)
    trainer = pl.Trainer(max_epochs=EPISODES, log_every_n_steps=1, devices=1)
    trainer.fit(model)
