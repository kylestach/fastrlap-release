from copy import deepcopy

import gym
from gym.wrappers.rescale_action import RescaleAction
from dm_control import composer
import numpy as np

from dmcgym.env import DMCGYM

from .task import CarTask
from .wrappers import KeysToStates, PermuteImage, RunningReturnInfo, ReplaceKey


PIXELS_STATES_KEYS = ['goal_relative', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc', 'car/wheel_speeds', 'car/steering_pos', 'car/steering_vel']
STATES_STATES_KEYS = ['goal_relative', 'car/body_pose_2d', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc', 'car/wheel_speeds', 'car/steering_pos', 'car/steering_vel']
# STATES_STATES_KEYS = ['goal_relative', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc']
# STATES_STATES_KEYS = ['goal_relative', 'car/body_pose_2d', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc']

def make_car_task_gym(*args, **kwargs):
    task = CarTask(*args, **kwargs)
    original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
    env = DMCGYM(original_env)
    env = ReplaceKey(env, 'car/realsense_camera', 'pixels')
    env = PermuteImage(env, 'pixels')
    env = KeysToStates(env, PIXELS_STATES_KEYS)
    env = RescaleAction(env, -np.ones(2), np.ones(2))
    env = RunningReturnInfo(env)
    return env

def make_car_task_gym_states(*args, **kwargs):
    task = CarTask(*args, include_camera=False, **kwargs)
    original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
    env = DMCGYM(original_env)
    env = KeysToStates(env, STATES_STATES_KEYS)
    env = RescaleAction(env, -np.ones(2), np.ones(2))
    env = RunningReturnInfo(env)
    return env


gym.register('offroad_sim/CarTask-v0', entry_point='offroad_sim_v2.envs:make_car_task_gym')
gym.register('offroad_sim/CarTask-states-v0', entry_point='offroad_sim_v2.envs:make_car_task_gym_states')

def make_dmc_env_record(rb, from_states=False):
    task = CarTask(include_camera=not from_states)
    env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

    data = {}

    def process_obs(observation):
        keys = STATES_STATES_KEYS if from_states else PIXELS_STATES_KEYS
        return {
            **{k: v for k, v in observation.items() if k not in ['car/realsense_camera']},
            'pixels': np.transpose(observation['car/realsense_camera'], (1, 2, 3, 0)),
            'states': np.concatenate([observation[k].reshape(-1) for k in keys], axis=0),
        }

    def process_action(action):
        action_bias = (env.action_spec().maximum + env.action_spec().minimum) / 2
        action_scale = (env.action_spec().maximum - env.action_spec().minimum) / 2
        return (action - action_bias) / action_scale

    def hook_before_step(physics, action, random_state):
        # Grab the observation and the action before the step
        nonlocal data, env
        if len(data):
            data['next_observations'] = process_obs(env._observation_updater.get_observation())
            data['rewards'] = env.task.get_reward(physics)
            done = env.task.should_terminate_episode(physics)
            data['dones'] = done
            data['masks'] = np.array([0.0 if done else 1.0])
            rb.insert(data)
            data = {}

        data['observations'] = deepcopy(process_obs(env._observation_updater.get_observation()))
        data['actions'] = process_action(action.copy())

        if env.task.should_terminate_episode(physics):
            data = {}

    env.add_extra_hook('before_step', hook_before_step)

    return env