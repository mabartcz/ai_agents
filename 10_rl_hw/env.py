import gymnasium as gym
from gymnasium import spaces
import numpy as np
from loguru import logger
from constants import GRID_SIZE, MAX_STEPS

class Grid2DEnv(gym.Env):
    def __init__(self):
        super(Grid2DEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = move up, 1 = move down, 2 = move left, 3 = move right
        self.action_space = spaces.Discrete(4)

        # Observations: the current position on the grid (row, column) + goal position (row, column)
        # State now includes both agent position and goal position: [agent_row, agent_col, goal_row, goal_col]
        self.observation_space = spaces.Box(low=0, high=GRID_SIZE-1, shape=(4,), dtype=np.float32)

        # Goal position will be set randomly in reset()
        self.goal_position = None

        # Limit episode length
        self.max_steps = MAX_STEPS

        # Initialize the state
        self.state = None
        self.episode_counter = 0
        self.step_counter = 0

    def _get_full_state(self):
        """Return the full state including agent position and goal position."""
        return np.concatenate([self.state, self.goal_position])

    def _normalize_state(self, state):
        """Normalize state for SB3: Convert grid position to range [0,1]."""
        return state.astype(np.float32) / (GRID_SIZE - 1)  # Divide by grid max index

    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Use np.clip() for boundary control
        new_state = self.state + np.array([[-1, 0], [1, 0], [0, -1], [0, 1]][action], dtype=np.int32)
        self.state = np.clip(new_state, 0, GRID_SIZE-1)

        # Calculate reward
        if np.array_equal(self.state, self.goal_position):
            reward = 1.0
            terminated = True # Natural terminal state - Goal reached
            truncated = False # Not a timeout
        elif self.step_counter >= self.max_steps:  # ✅ End if max steps reached
            reward = -1.0  # Mild penalty for running out of time
            terminated = False # Not a natural terminal state
            truncated = True  # Indicates timeout
        else:
            reward = -0.01 # Small penalty for each step
            terminated = False # Not a natural terminal state
            truncated = False # Not a timeout

        info = {}
        full_state = self._get_full_state()
        return full_state.astype(np.float32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)  # Initialize the random number generator with the seed

        # Increment episode counter
        self.episode_counter += 1

        # Reset step counter
        self.step_counter = 0  
        
        # Randomly set the goal position
        self.goal_position = np.random.randint(0, GRID_SIZE, size=(2,), dtype=np.int32)

        # Randomly initialize the agent's position, avoiding the goal position
        while True:
            self.state = np.random.randint(0, GRID_SIZE, size=(2,), dtype=np.int32)
            if not np.array_equal(self.state, self.goal_position):
                break

        # self.render()

        info = {}
        full_state = self._get_full_state()
        return full_state.astype(np.float32), info

    def render(self, mode="human"):
        # Render the environment (optional)
        grid = [["-"] * GRID_SIZE for _ in range(GRID_SIZE)]
        row, col = self.state
        grid[row][col] = "A"  # Agent's position
        gr, gc = self.goal_position
        grid[gr][gc] = "G"  # Goal position
        for row in grid:
            logger.info(" ".join(row))
        logger.info("")

    def close(self):
        # Cleanup any resources used by the environment (optional)
        pass
