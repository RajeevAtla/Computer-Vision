import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pytorch_lightning as pl

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

# PyTorch Lightning Module
class DQNLightning(pl.LightningModule):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon = EPSILON_START
        self.policy_net = DQN(state_shape, n_actions)
        self.target_net = DQN(state_shape, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.env = gym.make(ENV_NAME, render_mode=None)
        self.state_shape = state_shape
        self.save_hyperparameters()

    def forward(self, x):
        return self.policy_net(x)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.log("train_loss", loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def configure_optimizers(self):
        self.optim = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        return self.optim

    def train_dataloader(self):
        return DataLoader(self.replay_buffer, batch_size=BATCH_SIZE, shuffle=True)

    def on_train_epoch_end(self):
        if self.current_epoch % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# Main Training Loop
class DQNTrainer(pl.Trainer):
    def __init__(self, model):
        self.model = model
        self.env = model.env
        self.episodes = EPISODES

    def train(self):
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            state = np.moveaxis(state, -1, 0)

            for step in range(MAX_STEPS):
                action = self.model.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.moveaxis(next_state, -1, 0)

                self.model.replay_buffer.add(state, action, reward, next_state, done)

                if self.model.replay_buffer.size() > BATCH_SIZE:
                    batch = self.model.replay_buffer.sample(BATCH_SIZE)
                    self.model.training_step(batch, None)

                state = next_state
                total_reward += reward

                if done:
                    break

            self.model.epsilon = max(EPSILON_MIN, self.model.epsilon * EPSILON_DECAY)
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.model.epsilon:.2f}")

if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode=None)
    state_shape = env.observation_space.shape
    state_shape = (state_shape[2], state_shape[0], state_shape[1])
    n_actions = env.action_space.n

    model = DQNLightning(state_shape, n_actions)
    trainer = DQNTrainer(model)
    trainer.train()
