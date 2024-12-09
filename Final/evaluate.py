from stable_baselines3 import DQN
from pacman_env import PacmanEnv

# Load the trained model
model = DQN.load("dqn_pacman")

# Create the environment
env = PacmanEnv()
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    print(f"Reward: {reward}, Total Reward: {total_reward}")