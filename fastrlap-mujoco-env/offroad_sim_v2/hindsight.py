import numpy as np
import offroad_sim_v2.envs.utils as env_helpers

PIXELS_STATES_KEYS = ['goal_relative', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc', 'car/wheel_speeds', 'car/steering_pos', 'car/steering_vel']
STATES_STATES_KEYS = ['goal_relative', 'car/body_pose_2d', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc', 'car/wheel_speeds', 'car/steering_pos', 'car/steering_vel']

def relabel(batch, env, from_states=False):
    batch_size = batch['actions'].shape[0]

    observation = batch['observations']
    next_observation = batch['next_observations']
    future_observation = batch['future_observations']

    original_absolute = observation['goal_absolute']
    original_absolute_next = next_observation['goal_absolute']
    future_absolute = future_observation['car/body_pose_2d'][:, :2]
    random_absolute = observation['car/body_pose_2d'][:, :2] + np.random.normal(loc=0.0, scale=4.0, size=(batch_size, 2))

    def select(samples):
        indices = np.random.randint(0, len(samples), size=(batch_size, 1))
        return sum((indices == i) * sample for i, sample in enumerate(samples))

    goals_absolute = select([original_absolute, original_absolute, future_absolute, random_absolute])
    goals_absolute_next = goals_absolute.copy()

    goals_complete = np.linalg.norm(goals_absolute - observation['car/body_pose_2d'][:, :2], axis=-1) < env.goal_graph.goal_threshold
    goals_absolute_next[goals_complete] = original_absolute_next[goals_complete]

    keys = STATES_STATES_KEYS if from_states else PIXELS_STATES_KEYS

    goals_relative = env_helpers.batch_relative_goal_polarcoord(observation['car/body_pose_2d'], goals_absolute)
    goals_relative_next = env_helpers.batch_relative_goal_polarcoord(next_observation['car/body_pose_2d'], goals_absolute_next)

    observation['goal_absolute'] = goals_absolute
    observation['goal_relative'] = goals_relative
    next_observation['goal_absolute'] = goals_absolute_next
    next_observation['goal_relative'] = goals_relative_next

    new_states = np.concatenate([observation[k] for k in keys], axis=-1)
    new_next_states = np.concatenate([next_observation[k] for k in keys], axis=-1)

    observation['states'] = new_states
    next_observation['states'] = new_next_states

    reward = env.batch_compute_reward_from_observation(observation, batch['actions'], next_observation)

    return {
        **batch,
        'observations': observation,
        'next_observations': next_observation,
        'rewards': reward,
    }