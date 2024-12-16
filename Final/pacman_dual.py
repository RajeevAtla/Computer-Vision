import gym
from gym import spaces
import numpy as np

class PacmanEnv(gym.Env):
    def __init__(self):
        super(PacmanEnv, self).__init__()
        self.grid_size = 10
        self.action_space = spaces.Tuple((
            spaces.Discrete(4),  # Pacman actions: Up, Down, Left, Right
            spaces.Discrete(4)   # Ghost actions: Up, Down, Left, Right
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 5), dtype=np.float32),  # Pacman state
            spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 5), dtype=np.float32)   # Ghost state
        ))
        self.np_random = None
        self.seed()
        self.invincible_steps = 0
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
        num_pellets = 20
        for _ in range(num_pellets):
            x, y = self.np_random.integers(0, self.grid_size, 2)
            while [x, y] in [self.pacman_pos, self.ghost_pos, self.goal_pos]:
                x, y = self.np_random.integers(0, self.grid_size, 2)
            self.pellets[x, y] = 1

        # Initialize power-ups
        self.powerups = np.zeros((self.grid_size, self.grid_size))
        
        self._update_state()
        return self._get_obs()

    def step(self, actions):
        pacman_action, ghost_action = actions

        # Move Pacman
        if pacman_action == 0: self.pacman_pos[0] = max(self.pacman_pos[0] - 1, 0)  # Up
        elif pacman_action == 1: self.pacman_pos[0] = min(self.pacman_pos[0] + 1, self.grid_size - 1)  # Down
        elif pacman_action == 2: self.pacman_pos[1] = max(self.pacman_pos[1] - 1, 0)  # Left
        elif pacman_action == 3: self.pacman_pos[1] = min(self.pacman_pos[1] + 1, self.grid_size - 1)  # Right

        # Move Ghost
        if ghost_action == 0: self.ghost_pos[0] = max(self.ghost_pos[0] - 1, 0)  # Up
        elif ghost_action == 1: self.ghost_pos[0] = min(self.ghost_pos[0] + 1, self.grid_size - 1)  # Down
        elif ghost_action == 2: self.ghost_pos[1] = max(self.ghost_pos[1] - 1, 0)  # Left
        elif ghost_action == 3: self.ghost_pos[1] = min(self.ghost_pos[1] + 1, self.grid_size - 1)  # Right

        reward_pacman = -0.1
        reward_ghost = -0.1
        done = False

        # Pellet collection
        if self.pellets[self.pacman_pos[0], self.pacman_pos[1]] == 1:
            self.pellets[self.pacman_pos[0], self.pacman_pos[1]] = 0
            reward_pacman += 1

        # Power-up collection
        if self.powerups[self.pacman_pos[0], self.pacman_pos[1]] == 1:
            self.powerups[self.pacman_pos[0], self.pacman_pos[1]] = 0
            reward_pacman += 5
            self.invincible_steps = 10

        # Goal check
        if self.pacman_pos == self.goal_pos:
            reward_pacman += 10
            done = True

        # Ghost-Pacman interaction
        if self.pacman_pos == self.ghost_pos:
            if self.invincible_steps > 0:
                reward_pacman += 5
                reward_ghost -= 5
                self.ghost_pos = [self.grid_size - 1, self.grid_size - 1]  # Reset ghost
            else:
                reward_pacman -= 10
                reward_ghost += 10
                done = True

        # Decrease invincibility
        if self.invincible_steps > 0:
            self.invincible_steps -= 1

        # End game if all pellets are collected
        if np.sum(self.pellets) == 0:
            done = True

        self._update_state()
        return self._get_obs(), (reward_pacman, reward_ghost), done, {}

    def _get_obs(self):
        pacman_obs = np.zeros_like(self.state)
        pacman_obs[self.pacman_pos[0], self.pacman_pos[1], 0] = 1
        ghost_obs = np.zeros_like(self.state)
        ghost_obs[self.ghost_pos[0], self.ghost_pos[1], 2] = 1
        return pacman_obs, ghost_obs

    def render(self, mode="human"):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.pacman_pos[0], self.pacman_pos[1]] = 1  # Pacman
        grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Goal
        grid[self.ghost_pos[0], self.ghost_pos[1]] = 3  # Ghost
        grid[self.pellets == 1] = 4  # Pellets
        grid[self.powerups == 1] = 5  # Power-ups
        print(grid)
