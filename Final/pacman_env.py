import gym
from gym import spaces
import numpy as np

class PacmanEnv(gym.Env):
    def __init__(self):
        super(PacmanEnv, self).__init__()
        self.grid_size = 10
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size, 5), dtype=np.float32
        )  # 5 layers: Pacman, Goal, Ghost, Pellets, Power-ups
        self.np_random = None  # Placeholder for the random seed
        self.seed()  # Initialize the random seed
        self.invincible_steps = 0  # Steps Pacman remains invincible
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.pacman_pos = [0, 0]
        self.ghost_pos = [self.grid_size - 1, self.grid_size - 1]
        self.goal_pos = [self.grid_size - 1, 0]
        self.invincible_steps = 0

        # Initialize state layers
        self.state = np.zeros((self.grid_size, self.grid_size, 5))
        
        # Initialize pellets
        self.pellets = np.zeros((self.grid_size, self.grid_size))
        num_pellets = 20  # Number of pellets
        for _ in range(num_pellets):
            x, y = self.np_random.integers(0, self.grid_size, 2)
            while [x, y] in [self.pacman_pos, self.ghost_pos, self.goal_pos]:
                x, y = self.np_random.integers(0, self.grid_size, 2)
            self.pellets[x, y] = 1

        # Initialize power-ups
        self.powerups = np.zeros((self.grid_size, self.grid_size))
        
        self._update_state()
        return self.state

    def step(self, action):
        # Move Pacman
        if action == 0: self.pacman_pos[0] = max(self.pacman_pos[0] - 1, 0)  # Up
        elif action == 1: self.pacman_pos[0] = min(self.pacman_pos[0] + 1, self.grid_size - 1)  # Down
        elif action == 2: self.pacman_pos[1] = max(self.pacman_pos[1] - 1, 0)  # Left
        elif action == 3: self.pacman_pos[1] = min(self.pacman_pos[1] + 1, self.grid_size - 1)  # Right

        reward = -0.1  # Small penalty for each step
        done = False

        # Check for pellet collection
        if self.pellets[self.pacman_pos[0], self.pacman_pos[1]] == 1:
            self.pellets[self.pacman_pos[0], self.pacman_pos[1]] = 0  # Eat the pellet
            reward += 1  # Reward for eating a pellet

        # Check for power-up collection
        if self.powerups[self.pacman_pos[0], self.pacman_pos[1]] == 1:
            self.powerups[self.pacman_pos[0], self.pacman_pos[1]] = 0  # Consume power-up
            reward += 5  # Bonus reward
            self.invincible_steps = 10  # Pacman becomes invincible for 10 steps

        # Check for goal
        if self.pacman_pos == self.goal_pos:
            reward += 10
            done = True

        # Handle ghost interaction
        if self.pacman_pos == self.ghost_pos:
            if self.invincible_steps > 0:
                reward += 5  # Bonus for "eating" ghost while invincible
                self.ghost_pos = [self.grid_size - 1, self.grid_size - 1]  # Reset ghost position
            else:
                reward -= 10  # Penalty for getting caught
                done = True

        # Update ghost movement
        if self.invincible_steps == 0:  # Normal behavior
            if self.ghost_pos[0] < self.pacman_pos[0]:
                self.ghost_pos[0] += 1  # Move down
            elif self.ghost_pos[0] > self.pacman_pos[0]:
                self.ghost_pos[0] -= 1  # Move up

            if self.ghost_pos[1] < self.pacman_pos[1]:
                self.ghost_pos[1] += 1  # Move right
            elif self.ghost_pos[1] > self.pacman_pos[1]:
                self.ghost_pos[1] -= 1  # Move left

        # Reduce invincibility timer
        if self.invincible_steps > 0:
            self.invincible_steps -= 1

        # Spawn power-ups randomly
        if self.np_random.random() < 0.1:  # 10% chance to spawn a power-up each step
            self.spawn_powerup()

        # End game when all pellets are collected
        if np.sum(self.pellets) == 0:
            done = True

        # Update the state
        self._update_state()
        return self.state, reward, done, {}

    def spawn_powerup(self):
        x, y = self.np_random.integers(0, self.grid_size, 2)
        while [x, y] in [self.pacman_pos, self.ghost_pos, self.goal_pos] or self.pellets[x, y] == 1:
            x, y = self.np_random.integers(0, self.grid_size, 2)
        self.powerups[x, y] = 1

    def _update_state(self):
        self.state = np.zeros((self.grid_size, self.grid_size, 5))
        self.state[self.pacman_pos[0], self.pacman_pos[1], 0] = 1  # Pacman
        self.state[self.goal_pos[0], self.goal_pos[1], 1] = 1  # Goal
        self.state[self.ghost_pos[0], self.ghost_pos[1], 2] = 1  # Ghost
        self.state[..., 3] = self.pellets  # Pellets
        self.state[..., 4] = self.powerups  # Power-ups

    def render(self, mode="human"):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.pacman_pos[0], self.pacman_pos[1]] = 1  # Pacman
        grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Goal
        grid[self.ghost_pos[0], self.ghost_pos[1]] = 3  # Ghost
        grid[self.pellets == 1] = 4  # Pellets
        grid[self.powerups == 1] = 5  # Power-ups
        print(grid)