import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import setup
from loguru import logger
from constants import GRID_SIZE
from tqdm import tqdm

# Create environment
env = gym.make("Grid2DEnv-v0")

# Q-learning parameters
alpha = 0.2  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.3  # Exploration rate
episodes = 10000  # Number of training episodes

# Q-table: GRID_SIZE x GRID_SIZE (agent position) x GRID_SIZE x GRID_SIZE (goal position) x 4 actions
# q_table[agent_row, agent_col, goal_row, goal_col, action]
q_table = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE, 4))

# Progress tracking
steps_per_episode = []  # Track number of steps taken in each episode


def choose_action(state):
    """Epsilon-greedy action selection"""
    agent_row, agent_col, goal_row, goal_col = map(int, state)  # Ensure indices are integers

    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Random action

    return np.argmax(q_table[agent_row, agent_col, goal_row, goal_col])  # Greedy action


def train_q_learning():
    """Q-learning training loop (off-policy)"""
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state, _ = env.reset()
        state = state.astype(int)  # Convert to integer

        done = False
        step_count = 0  # Initialize step counter for this episode

        while not done:
            action = choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.astype(int)  # Convert to integer

            agent_row, agent_col, goal_row, goal_col = state
            next_agent_row, next_agent_col, next_goal_row, next_goal_col = next_state

            # ----------------------------------------------
            # Q-learning update rule (off-policy)
            # Note: goal position doesn't change during an episode, so next_goal == goal
            q_table[agent_row, agent_col, goal_row, goal_col, action] += alpha * (
                reward
                + gamma * np.max(q_table[next_agent_row, next_agent_col, next_goal_row, next_goal_col, :])
                - q_table[agent_row, agent_col, goal_row, goal_col, action]
            )

            state = next_state  # Move to next state
            step_count += 1  # Increment step counter
            # ----------------------------------------------

            done = terminated or truncated

        steps_per_episode.append(step_count)  # Track steps for this episode

    np.save("q_learning_q_table.npy", q_table)
    logger.success("Q-learning training complete. Q-table saved!")


def plot_training_progress():
    """Plot the training progress showing steps per episode"""
    plt.figure(figsize=(12, 8))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Steps per episode
    ax1.plot(steps_per_episode, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps per Episode')
    ax1.set_title('Q-Learning Training Progress: Steps per Episode')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Moving average of steps (smoothed progress)
    window_size = 50
    if len(steps_per_episode) >= window_size:
        moving_avg = np.convolve(steps_per_episode, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(steps_per_episode)), moving_avg, 'r-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Moving Average Steps')
        ax2.set_title(f'Moving Average of Steps per Episode (Window size: {window_size})')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics
    logger.info(f"\nTraining Statistics:")
    logger.info(f"Total episodes: {len(steps_per_episode)}")
    logger.info(f"Average steps per episode: {np.mean(steps_per_episode):.2f}")
    logger.info(f"Min steps: {np.min(steps_per_episode)}")
    logger.info(f"Max steps: {np.max(steps_per_episode)}")
    logger.info(f"Final 100 episodes average: {np.mean(steps_per_episode[-100:]):.2f}")


if __name__ == "__main__":
    train_q_learning()
    plot_training_progress()
    env.close()
