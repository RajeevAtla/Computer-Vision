import gym
class PacmanEnv(gym.Env):
    def __init__(self):
        super(PacmanEnv, self).__init__()
        self.grid_size = 10
        
        # Action spaces for Pacman and Ghost
        self.action_space = spaces.Tuple((
            spaces.Discrete(4),  # Pacman: Up, Down, Left, Right
            spaces.Discrete(4)   # Ghost: Up, Down, Left, Right
        ))
        
        # Observation space for both agents
        self.observation_space = spaces.Dict({
            "pacman": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 5), dtype=np.float32),
            "ghost": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 5), dtype=np.float32),
        })

        self.seed()
        self.reset()

    def step(self, actions):
        pacman_action, ghost_action = actions

        # Pacman movement
        if pacman_action == 0: self.pacman_pos[0] = max(self.pacman_pos[0] - 1, 0)
        elif pacman_action == 1: self.pacman_pos[0] = min(self.pacman_pos[0] + 1, self.grid_size - 1)
        elif pacman_action == 2: self.pacman_pos[1] = max(self.pacman_pos[1] - 1, 0)
        elif pacman_action == 3: self.pacman_pos[1] = min(self.pacman_pos[1] + 1, self.grid_size - 1)

        # Ghost movement
        if ghost_action == 0: self.ghost_pos[0] = max(self.ghost_pos[0] - 1, 0)
        elif ghost_action == 1: self.ghost_pos[0] = min(self.ghost_pos[0] + 1, self.grid_size - 1)
        elif ghost_action == 2: self.ghost_pos[1] = max(self.ghost_pos[1] - 1, 0)
        elif ghost_action == 3: self.ghost_pos[1] = min(self.ghost_pos[1] + 1, self.grid_size - 1)

        # Compute rewards
        pacman_reward, ghost_reward = -0.1, -0.1  # Default penalties for each step

        if self.pacman_pos == self.ghost_pos:  # Ghost catches Pacman
            pacman_reward -= 10
            ghost_reward += 10
            done = True
        else:
            done = False

        # Update state and return observations for both agents
        self._update_state()
        return {
            "pacman": self.state,
            "ghost": self.state,
        }, {
            "pacman": pacman_reward,
            "ghost": ghost_reward,
        }, done, {}

    def reset(self):
        # Reset positions, rewards, and state
        self.pacman_pos = [0, 0]
        self.ghost_pos = [self.grid_size - 1, self.grid_size - 1]
        self._update_state()
        return {
            "pacman": self.state,
            "ghost": self.state,
        }