import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gym import Env, spaces
from PIL import Image, ImageDraw, ImageFont
import time
import matplotlib.pyplot as plt

# Adjusted Hyperparameters
LEARNING_RATE = 5e-5
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 20
EPISODES = 500
MAX_STEPS = 500

class ObstacleGhostPacmanEnv(Env):
   def __init__(self):
      super(ObstacleGhostPacmanEnv, self).__init__()
      self.grid_size = 20
      self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
      self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

      self.initial_energy = 100
      self.energy_per_pill = 20
      self.energy_step_cost = 1
      self.reset()

   def reset(self):
      self.pacman_pos = [0, 0]
      self.energy = self.initial_energy

      # Two ghosts
      self.ghost_positions = [
         [self.grid_size - 1, self.grid_size - 1],
         [0, self.grid_size - 1],
      ]

      # Generate obstacles as line segments
      self.obstacles = np.zeros((self.grid_size, self.grid_size))
      self.generate_line_obstacles(num_segments=8, min_length=3, max_length=6)

      # Normal pellets (white)
      self.pellets = np.zeros((self.grid_size, self.grid_size))
      num_pellets = 20
      for _ in range(num_pellets):
         x, y = np.random.randint(0, self.grid_size, size=2)
         # Avoid placing on obstacles, pacman start, or ghosts
         while (self.obstacles[x, y] == 1 or
                  [x, y] == self.pacman_pos or
                  [x, y] in self.ghost_positions):
               x, y = np.random.randint(0, self.grid_size, size=2)
         self.pellets[x, y] = 1

      # Energy pills (blue)
      self.energy_pills = np.zeros((self.grid_size, self.grid_size))
      num_energy_pills = 10
      for _ in range(num_energy_pills):
         x, y = np.random.randint(0, self.grid_size, size=2)
         # Avoid placing on obstacles, ghosts, pacman, pellets
         while (self.obstacles[x, y] == 1 or
                  [x, y] == self.pacman_pos or
                  [x, y] in self.ghost_positions or
                  self.pellets[x, y] == 1):
               x, y = np.random.randint(0, self.grid_size, size=2)
         self.energy_pills[x, y] = 1

      return self.render_observation()

   def generate_line_obstacles(self, num_segments=8, min_length=3, max_length=6):
      """
      Generate obstacles as small line segments. Each segment is either horizontal or vertical.
      """
      for _ in range(num_segments):
         # Random orientation
         horizontal = random.choice([True, False])
         length = random.randint(min_length, max_length)

         if horizontal:
               row = np.random.randint(0, self.grid_size)
               col_start = np.random.randint(0, self.grid_size - length)
               for c in range(col_start, col_start + length):
                  if [row, c] != self.pacman_pos and [row, c] not in self.ghost_positions:
                     self.obstacles[row, c] = 1
         else:
               col = np.random.randint(0, self.grid_size)
               row_start = np.random.randint(0, self.grid_size - length)
               for r in range(row_start, row_start + length):
                  if [r, col] != self.pacman_pos and [r, col] not in self.ghost_positions:
                     self.obstacles[r, col] = 1

   def step(self, action, stepnum):
      print(f'{stepnum}', end=' ')
      old_pos = self.pacman_pos.copy()

      # Move Pacman
      if action == 0:  # Up
         self.pacman_pos[0] = max(self.pacman_pos[0] - 1, 0)
      elif action == 1:  # Down
         self.pacman_pos[0] = min(self.pacman_pos[0] + 1, self.grid_size - 1)
      elif action == 2:  # Left
         self.pacman_pos[1] = max(self.pacman_pos[1] - 1, 0)
      elif action == 3:  # Right
         self.pacman_pos[1] = min(self.pacman_pos[1] + 1, self.grid_size - 1)

      reward = -0.1

      # Handle obstacle collision
      if self.obstacles[self.pacman_pos[0], self.pacman_pos[1]] == 1:
         self.pacman_pos = old_pos
         reward -= 0.5

      # Check for pellet collection
      if self.pellets[self.pacman_pos[0], self.pacman_pos[1]] == 1:
         self.pellets[self.pacman_pos[0], self.pacman_pos[1]] = 0
         reward += 5

      # Check for energy pill collection
      if self.energy_pills[self.pacman_pos[0], self.pacman_pos[1]] == 1:
         self.energy_pills[self.pacman_pos[0], self.pacman_pos[1]] = 0
         reward += 5
         self.energy += self.energy_per_pill

      # Decrease energy by step cost
      self.energy -= self.energy_step_cost

      # Move ghosts
      new_ghost_positions = []
      ghost_collision = False
      for ghost_pos in self.ghost_positions:
         ghost_moves = []
         xg, yg = ghost_pos
         if xg > 0 and self.obstacles[xg - 1, yg] == 0:
               ghost_moves.append((xg - 1, yg))
         if xg < self.grid_size - 1 and self.obstacles[xg + 1, yg] == 0:
               ghost_moves.append((xg + 1, yg))
         if yg > 0 and self.obstacles[xg, yg - 1] == 0:
               ghost_moves.append((xg, yg - 1))
         if yg < self.grid_size - 1 and self.obstacles[xg, yg + 1] == 0:
               ghost_moves.append((xg, yg + 1))

         if not ghost_moves:
               new_pos = ghost_pos
         else:
               if random.random() < 0.2:
                  new_pos = random.choice(ghost_moves)
               else:
                  new_pos = min(
                     ghost_moves,
                     key=lambda move: abs(move[0] - self.pacman_pos[0]) + abs(move[1] - self.pacman_pos[1])
                  )
         new_ghost_positions.append(list(new_pos))

      self.ghost_positions = new_ghost_positions

      # Check ghost collision
      for gpos in self.ghost_positions:
         if self.pacman_pos == gpos:
               reward -= 100
               ghost_collision = True

      # Check energy depletion
      done = False
      if self.energy <= 0:
         done = True
         print(f"Pacman ran out of energy on step number {stepnum}!")

      # If all pellets and energy pills are gone, episode ends
      if np.sum(self.pellets) == 0 and np.sum(self.energy_pills) == 0:
         done = True

      if ghost_collision:
         print(f"got caught on step number {stepnum}!")
         done = True

      info = {}
      if ghost_collision:
         info['ghost_collision'] = True

      if done and not ghost_collision and self.energy <= 0:
         # Energy run out scenario already printed
         pass

      return self.render_observation(), reward, done, info

   def render_observation(self):
      img = Image.new("RGB", (84, 84), "black")
      draw = ImageDraw.Draw(img)

      # Obstacles
      for x, y in np.argwhere(self.obstacles == 1):
         draw.rectangle([(x * 4, y * 4), (x * 4 + 3, y * 4 + 3)], fill="gray")

      # Pellets (white)
      for x, y in np.argwhere(self.pellets == 1):
         draw.rectangle([(x * 4 + 1, y * 4 + 1), (x * 4 + 2, y * 4 + 2)], fill="white")

      # Energy pills (blue)
      for x, y in np.argwhere(self.energy_pills == 1):
         draw.rectangle([(x * 4 + 1, y * 4 + 1), (x * 4 + 2, y * 4 + 2)], fill="blue")

      # Pacman (yellow)
      px, py = self.pacman_pos
      draw.rectangle([(px * 4, py * 4), (px * 4 + 3, py * 4 + 3)], fill="yellow")

      # Ghosts (red)
      for gx, gy in self.ghost_positions:
         draw.rectangle([(gx * 4, gy * 4), (gx * 4 + 3, gy * 4 + 3)], fill="red")

      return np.array(img)

   def render_frame(self):
      return self.render_observation()

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

class ReplayBuffer:
   def __init__(self, capacity):
      self.buffer = deque(maxlen=capacity)

   def add(self, state, action, reward, next_state, done):
      self.buffer.append((state, action, reward, next_state, done))

   def sample(self, batch_size):
      return random.sample(self.buffer, batch_size)

   def size(self):
      return len(self.buffer)

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

      return loss.item()

   def update_target_net(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())

def plot_and_save(metric, ma_metric, title, ylabel, filename):
   plt.figure()
   plt.plot(metric, label='Raw')
   plt.plot(ma_metric, label='Moving Avg', linewidth=2)
   plt.xlabel('Episode')
   plt.ylabel(ylabel)
   plt.title(title)
   plt.legend()
   plt.savefig(filename, dpi=300)
   plt.close()

def train():
   env = ObstacleGhostPacmanEnv()
   state_shape = (3, 84, 84)
   n_actions = env.action_space.n
   agent = DQNAgent(state_shape, n_actions)

   # Metrics storage
   episode_rewards = []
   episode_lengths = []
   episode_losses = []
   episode_avg_qs = []
   episode_times = []

   # Moving average window
   ma_window = 10

   def moving_average(data, window=ma_window):
      ma = []
      for i in range(len(data)):
         if i < window:
               ma.append(np.mean(data[:i+1]))
         else:
               ma.append(np.mean(data[i-window+1:i+1]))
      return ma

   for episode in range(EPISODES):
      start_time = time.time()
      state = env.reset()
      total_reward = 0
      total_loss = 0
      total_q = 0
      steps = 0

      # Convert to CHW
      state = state.transpose(2, 0, 1)

      for step in range(MAX_STEPS):
         # Get Q-values and select action
         state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
         q_values = agent.policy_net(state_t)
         action = agent.select_action(state)

         chosen_q = q_values[0, action].item()
         total_q += chosen_q

         next_state, reward, done, info = env.step(action, step)
         next_state = next_state.transpose(2, 0, 1)

         loss = agent.train()  # Train returns loss, if any
         if loss is not None:
               total_loss += loss

         agent.replay_buffer.add(state, action, reward, next_state, done)

         state = next_state
         total_reward += reward
         steps += 1

         if done:
               print("Ghost caught Pacman or no pellets/energy pills remain or ran out of energy. Ending this episode.")
               break  # End of episode

      # Update epsilon and target net periodically
      agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
      if episode % TARGET_UPDATE == 0:
         agent.update_target_net()

      end_time = time.time()
      episode_time = end_time - start_time

      # Record metrics for this episode
      episode_rewards.append(total_reward)
      episode_lengths.append(steps)
      episode_times.append(episode_time)
      avg_loss = total_loss / steps if steps > 0 else 0
      episode_losses.append(avg_loss)
      avg_q = total_q / steps if steps > 0 else 0
      episode_avg_qs.append(avg_q)

      # Compute current moving averages
      ma_rewards = moving_average(episode_rewards, ma_window)
      ma_lengths = moving_average(episode_lengths, ma_window)
      ma_losses = moving_average(episode_losses, ma_window)
      ma_qs = moving_average(episode_avg_qs, ma_window)
      ma_times = moving_average(episode_times, ma_window)

      print(f"Episode {episode + 1}/{EPISODES}, "
            f"Reward: {total_reward:.2f}, MA(Reward): {ma_rewards[-1]:.2f}, "
            f"Length: {steps}, MA(Length): {ma_lengths[-1]:.2f}, "
            f"Loss: {avg_loss:.4f}, MA(Loss): {ma_losses[-1]:.4f}, "
            f"AvgQ: {avg_q:.2f}, MA(AvgQ): {ma_qs[-1]:.2f}, "
            f"Epsilon: {agent.epsilon:.2f}, "
            f"Time: {episode_time:.2f}s, MA(Time): {ma_times[-1]:.2f}s")

   env.close()

   # Plot and save metrics
   ma_rewards = moving_average(episode_rewards, ma_window)
   plot_and_save(episode_rewards, ma_rewards, 'Rewards per Episode', 'Reward', 'rewards.jpg')

   ma_lengths = moving_average(episode_lengths, ma_window)
   plot_and_save(episode_lengths, ma_lengths, 'Episode Lengths', 'Length', 'lengths.jpg')

   ma_losses = moving_average(episode_losses, ma_window)
   plot_and_save(episode_losses, ma_losses, 'Average Loss per Episode', 'Loss', 'losses.jpg')

   ma_qs = moving_average(episode_avg_qs, ma_window)
   plot_and_save(episode_avg_qs, ma_qs, 'Average Q-values per Episode', 'Avg Q-value', 'avg_q_values.jpg')

   ma_times = moving_average(episode_times, ma_window)
   plot_and_save(episode_times, ma_times, 'Episode Times', 'Time (s)', 'times.jpg')

   return agent

def save_sample_game(agent, env, output_path="sample_game.gif"):
   """
   Simulates a game using the trained agent and saves it as a GIF.
   Stops recording immediately when the ghost collision or energy depletion occurs.
   Shows only Step and Energy, with smaller font.
   """
   state = env.reset()
   state = state.transpose(2, 0, 1)  # Convert HWC to CHW
   frames = []
   done = False
   step = 0

   # Smaller font size
   try:
      font = ImageFont.truetype("arial.ttf", 6)
   except:
      # Fallback to default font if arial.ttf isn't available
      font = ImageFont.load_default()

   while not done:
      step += 1
      frame = env.render_frame()
      frame_image = Image.fromarray(frame)
      draw = ImageDraw.Draw(frame_image)

      # Add step and energy info with smaller font
      draw.text((2, 2), f"Step: {step}, Energy: {env.energy}", fill="white", font=font)
      frames.append(frame_image)

      action = agent.select_action(state)
      next_state, reward, done, info = env.step(action, step)
      next_state = next_state.transpose(2, 0, 1)
      state = next_state

      if done and info.get('ghost_collision', False):
         # Ghost collision
         final_frame = frames[-1].copy()
         draw_final = ImageDraw.Draw(final_frame)
         draw_final.text((2, 20), "Ghost collision!", fill="red", font=font)
         frames[-1] = final_frame
         break

      if done and env.energy <= 0:
         # Energy depletion
         final_frame = frames[-1].copy()
         draw_final = ImageDraw.Draw(final_frame)
         draw_final.text((2, 20), "Energy depleted!", fill="red", font=font)
         frames[-1] = final_frame
         break

   frames[0].save(
      output_path,
      save_all=True,
      append_images=frames[1:],
      duration=100,
      loop=0
   )
   print(f"Sample game saved to {output_path}")

if __name__ == "__main__":
   trained_agent = train()
   test_env = ObstacleGhostPacmanEnv()
   save_sample_game(trained_agent, test_env, output_path="sample_game.gif")
