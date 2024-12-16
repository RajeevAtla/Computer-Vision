from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from pacman_env import PacmanEnv
import numpy as np
import matplotlib.pyplot as plt

# Initialize environment
env = PacmanEnv()

# Wrap the environment for Stable-Baselines3
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize separate models for Pacman and Ghost
pacman_model = DQN("MlpPolicy", vec_env, verbose=1, gamma=0.99)
ghost_model = DQN("MlpPolicy", vec_env, verbose=1, gamma=0.99)

# Training parameters
pacman_losses, ghost_losses = [], []
n_steps = 10000

# Training loop
for step in range(n_steps):
    # Train Pacman
    pacman_model.learn(total_timesteps=1, log_interval=None)
    pacman_loss = compute_dqn_loss(pacman_model, pacman_model.replay_buffer)
    pacman_losses.append(pacman_loss)

    # Train Ghost
    ghost_model.learn(total_timesteps=1, log_interval=None)
    ghost_loss = compute_dqn_loss(ghost_model, ghost_model.replay_buffer)
    ghost_losses.append(ghost_loss)

    if step % 100 == 0:
        print(f"Step {step}: Pacman Loss = {pacman_loss}, Ghost Loss = {ghost_loss}")

# Save and plot losses
np.save("pacman_losses.npy", pacman_losses)
np.save("ghost_losses.npy", ghost_losses)

# Plot Pacman and Ghost losses
plt.figure(figsize=(12, 6))
plt.plot(pacman_losses, label="Pacman Loss")
plt.plot(ghost_losses, label="Ghost Loss")
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("Loss Over Time for Pacman and Ghost")
plt.legend()
plt.grid()
plt.show()