from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from pacman_env import PacmanEnv
import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3.dqn.policies import QNetwork

import torch

def compute_dqn_loss(model, replay_buffer):
    # Sample a batch from the replay buffer
    batch = model.replay_buffer.sample(32)  # Batch size
    observations = torch.tensor(batch.observations, dtype=torch.float32).clone().detach()
    actions = torch.tensor(batch.actions, dtype=torch.long).clone().detach()
    rewards = torch.tensor(batch.rewards, dtype=torch.float32).clone().detach()
    next_observations = torch.tensor(batch.next_observations, dtype=torch.float32).clone().detach()
    dones = torch.tensor(batch.dones, dtype=torch.float32).clone().detach()



    # Compute current Q-values
    q_values = model.policy.q_net(observations)  # Shape: [batch_size, num_actions]

    # Ensure actions have the correct shape
    actions = actions.view(-1, 1)  # Reshape to [batch_size, 1]

    # Gather Q-values corresponding to chosen actions
    q_values = q_values.gather(1, actions).squeeze()  # Shape: [batch_size]

    # Compute target Q-values
    next_q_values = model.policy.q_net(next_observations).max(dim=1)[0]  # Max Q-value for next state
    target_q_values = rewards + (1 - dones) * model.gamma * next_q_values

    # Compute Mean Squared Error loss
    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
    return loss.item()

# Create and wrap the environment
env = PacmanEnv()
vec_env = make_vec_env(lambda: env, n_envs=1)
env.render()
# Initialize the model
model = DQN("MlpPolicy", vec_env, verbose=1)

# Initialize an array to store losses
losses = []


# Train step by step
for _ in range(10000):  # Number of timesteps
    model.learn(total_timesteps=1, log_interval=None)
    
    # Compute and log the loss manually
    loss = compute_dqn_loss(model, model.replay_buffer)
    losses.append(loss)

# Save losses for plotting
np.save("losses.npy", losses)
# Save the losses
np.save("losses.npy", losses)

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(losses, label="Loss")
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.legend()
plt.grid()
plt.show()