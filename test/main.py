import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

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
        
        # Compute the size of the flattened input to the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)  # Create a dummy input with the same shape as the input to the network
            conv_output = self.conv(dummy_input)
            self.conv_output_size = conv_output.numel()  # Total number of elements in the output

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 512),  # Use computed size
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv(x / 255.0)  # Normalize pixel values
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
        self.policy_net = DQN(state_shape, n_actions)  # No .cuda()
        self.target_net = DQN(state_shape, n_actions)  # No .cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

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

# Main training loop
def train():
    env = gym.make(ENV_NAME, render_mode="human")
    state_shape = env.observation_space.shape
    state_shape = (state_shape[2], state_shape[0], state_shape[1])  # Convert (H, W, C) to (C, H, W)
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions)

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        state = np.moveaxis(state, -1, 0)  # Convert HWC to CHW

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.moveaxis(next_state, -1, 0)  # Convert HWC to CHW

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    env.close()

if __name__ == "__main__":
    train()
