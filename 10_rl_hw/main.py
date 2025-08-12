import numpy as np
import gymnasium as gym
import setup
from loguru import logger
from constants import GRID_SIZE

if __name__ == "__main__":

    # Create environment
    env = gym.make("Grid2DEnv-v0")
    state, _ = env.reset()

    state = state.astype(int)
    logger.info(f"Initial state: {state}")

    # Load trained Q-table
    q_table = np.load("q_learning_q_table.npy")
    logger.info("=========================================")
    logger.info("   Q-Table (State-Action Values)")
    logger.info(f"Dimensions: (Agent Rows = {GRID_SIZE}, Agent Cols = {GRID_SIZE}, Goal Rows = {GRID_SIZE}, Goal Cols = {GRID_SIZE}, Actions = 4)")
    logger.info(
        "Each cell stores the value of taking an action (Up=0, Down=1, Left=2, Right=3) in that state.\n"
        "State format: [agent_row, agent_col, goal_row, goal_col]\n"
    )

    # Extract current goal position for display
    agent_row, agent_col, goal_row, goal_col = state
    logger.info(f"Current goal position: ({goal_row}, {goal_col})")
    logger.info(f"Q-values for states with current goal position:")

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rounded_values = np.round(
                q_table[row, col, goal_row, goal_col, :], 2
            )  # Round values to 2 decimal places
            logger.info(f"Agent at ({row},{col}) -> Goal at ({goal_row},{goal_col}): {rounded_values}")

    logger.info("=========================================\n")
    logger.info("=========================================")
    logger.info("========== Testing the Agent ============")
    logger.info(f"Initial State: Agent at ({agent_row},{agent_col}), Goal at ({goal_row},{goal_col})")

    env.render()

    done = False
    while not done:
        logger.info("----------------------------")
        logger.info(f"Step {env.unwrapped.step_counter}")
        logger.info("----------------------------")

        logger.debug(f"State 1a {state}")
        agent_row, agent_col, goal_row, goal_col = map(int, state)  # Get current position and goal
        logger.debug(f"State 1b Agent: ({agent_row}, {agent_col}), Goal: ({goal_row}, {goal_col})")

        # Select the best action based on Q-table
        action = np.argmax(q_table[agent_row, agent_col, goal_row, goal_col, :])

        # Print Q-values for the current state
        logger.info(
            f"Q-values:  Up (0): {q_table[agent_row, agent_col, goal_row, goal_col, 0]:.2f} | Down (1): {q_table[agent_row, agent_col, goal_row, goal_col, 1]:.2f} | Left (2): {q_table[agent_row, agent_col, goal_row, goal_col, 2]:.2f} | Right (3): {q_table[agent_row, agent_col, goal_row, goal_col, 3]:.2f}"
        )

        # Show the chosen action
        action_text = ["Up (0)", "Down (1)", "Left (2)", "Right (3)"][action]
        logger.info(f"Chosen Action: {action_text}")

        # Take action
        state, reward, terminated, truncated, info = env.step(action)

        # The next state
        logger.debug(f"Next State {state}")
        state = state.astype(int)
        next_agent_row, next_agent_col, next_goal_row, next_goal_col = state
        logger.debug(f"Next State Agent: ({next_agent_row}, {next_agent_col}), Goal: ({next_goal_row}, {next_goal_col})")

        # Render environment
        env.render()

        # Stop if the episode ends
        done = terminated or truncated

    env.close()
    logger.info("=========================================")
    logger.info("\n========== Testing Complete ===========")
