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


def compute_custom_rewards(state, reward, done, agent_type, info=None, last_action=None, current_action=None):
    """
    Compute custom rewards based on the state, reward, and agent type.

    :param state: Current state of the environment.
    :param reward: Default reward from the environment.
    :param done: Whether the episode is done.
    :param agent_type: Type of the agent ('pacman' or 'ghost').
    :param info: Additional information from the environment (optional).
    :param last_action: The last action taken by the agent.
    :param current_action: The current action taken by the agent.
    :return: Custom reward for the agent.
    """
    custom_reward = reward

    if agent_type == 'pacman':
        if reward > 0:
            custom_reward += 10  # Bonus for eating dots or pellets
        if done and reward > 0:
            custom_reward += 50  # Bonus for successfully clearing a level
        
        # Negative reward for encountering a ghost
        if info and 'ghost_collision' in info and info['ghost_collision']:
            custom_reward -= 100

        # Slight penalty for repeated motions
        if last_action is not None and current_action == last_action:
            custom_reward -= 1  # Penalize repeated actions slightly

    elif agent_type == 'ghost':
        custom_reward = -reward  # Ghosts minimize Pacman's reward

        # Bonus for catching Pacman
        if info and 'ghost_collision' in info and info['ghost_collision']:
            custom_reward += 100

        # Encourage chasing Pacman if close
        if info and 'distance_to_pacman' in info and info['distance_to_pacman'] < 5:
            custom_reward += 10

    return custom_reward

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame.
        :param env: Environment to wrap.
        :param skip: Number of frames to skip.
        """
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """
        Repeat an action for `skip` frames.
        """
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info
    
# Main training loop
import matplotlib.pyplot as plt

def train_with_metrics():
    env = gym.make(ENV_NAME, render_mode=None)
    env = FrameSkip(env, skip=4)
    state_shape = env.observation_space.shape
    state_shape = (state_shape[2], state_shape[0], state_shape[1])
    n_actions = env.action_space.n

    pacman_agent = DQNAgent(state_shape, n_actions)
    ghost_agent = DQNAgent(state_shape, n_actions)

    # Metrics
    pacman_rewards = []
    ghost_rewards = []
    pacman_losses = []
    ghost_losses = []
    epsilons = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward_pacman = 0
        total_reward_ghost = 0
        state = np.moveaxis(state, -1, 0)

        pacman_loss = 0
        ghost_loss = 0
        steps = 0

        for step in range(MAX_STEPS):
            # Pacman action
            pacman_action = pacman_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(pacman_action)

            # Compute custom rewards for Pacman
            pacman_reward = compute_custom_rewards(state, reward, done, agent_type='pacman')

            # Ghost action (dummy action for this example)
            ghost_action = ghost_agent.select_action(state)
            ghost_reward = compute_custom_rewards(state, reward, done, agent_type='ghost')

            next_state = np.moveaxis(next_state, -1, 0)

            # Store experiences
            pacman_agent.replay_buffer.add(state, pacman_action, pacman_reward, next_state, done)
            ghost_agent.replay_buffer.add(state, ghost_action, ghost_reward, next_state, done)

            # Train agents and accumulate losses
            pacman_agent.train()
            ghost_agent.train()

            state = next_state
            total_reward_pacman += pacman_reward
            total_reward_ghost += ghost_reward

            steps += 1
            if done:
                break

        # Update metrics
        pacman_rewards.append(total_reward_pacman)
        ghost_rewards.append(total_reward_ghost)
        pacman_losses.append(pacman_loss / steps if steps > 0 else 0)
        ghost_losses.append(ghost_loss / steps if steps > 0 else 0)
        epsilons.append(pacman_agent.epsilon)

        pacman_agent.epsilon = max(EPSILON_MIN, pacman_agent.epsilon * EPSILON_DECAY)
        ghost_agent.epsilon = max(EPSILON_MIN, ghost_agent.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            pacman_agent.update_target_net()
            ghost_agent.update_target_net()

        print(f"Episode {episode + 1}, Pacman Total Reward: {total_reward_pacman}, Ghost Total Reward: {total_reward_ghost}, Epsilon: {pacman_agent.epsilon:.2f}")

    env.close()



if __name__ == "__main__":
    train_with_metrics()
