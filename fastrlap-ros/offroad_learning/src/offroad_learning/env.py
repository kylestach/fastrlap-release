import numpy as np

def batch_reward(state, action, next_state):
    goal_projection = np.sum(state["goal_relative"] * state["relative_linear_velocity"], axis=-1)

    return goal_projection