from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from pacman_env import PacmanEnv

# Initialize the environment
env = PacmanEnv()

# Separate policies for Pacman and Ghost
pacman_model = DQN("MlpPolicy", env, verbose=1)
ghost_model = DQN("MlpPolicy", env, verbose=1)

# Train both models
pacman_timesteps = 10000
ghost_timesteps = 10000

print("Training Pacman Agent...")
pacman_model.learn(total_timesteps=pacman_timesteps)

print("Training Ghost Agent...")
ghost_model.learn(total_timesteps=ghost_timesteps)

# Save models
pacman_model.save("pacman_model")
ghost_model.save("ghost_model")
