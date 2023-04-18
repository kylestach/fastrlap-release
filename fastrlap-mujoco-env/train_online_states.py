#! /usr/bin/env python
import os
import pickle
import time

import gym
import gym.spaces as spaces
from gym.wrappers.time_limit import TimeLimit
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
import numpy as np

from jaxrl5.agents import SACLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.evaluation import evaluate

# Offroad stuff
import warnings

warnings.filterwarnings("ignore")
import offroad_sim_v2.envs
from offroad_sim_v2.hindsight import relabel
from flax.core.frozen_dict import FrozenDict

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'offroad_sim/CarTask-states-v0',
                    'Environment name.')
flags.DEFINE_string('comment', '', 'Comment for W&B')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('expert_replay_buffer', '',
                    '(Optional) Expert replay buffer pickle file.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 100, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of training steps to start training.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Capacity of the replay buffer.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('save_buffer', True, 'Save the replay buffer.')
flags.DEFINE_integer('utd_ratio', 8, 'Update-to-data ratio')
config_flags.DEFINE_config_file(
    'config',
    'configs/redq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def filter_obs(obs):
    return obs['states']


def filter_obs_space(obs_space):
    return obs_space['states']


def filter_batch(obs):
    return FrozenDict({
        'observations': filter_obs(obs['observations']),
        'actions': obs['actions'],
        'next_observations': filter_obs(obs['next_observations']),
        'rewards': obs['rewards'],
        'dones': obs['dones'],
        'masks': obs['masks'],
    })


class SelectObs(gym.ObservationWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self.key = key
        self.observation_space = env.observation_space[self.key]

    def observation(self, obs):
        return obs[self.key]


def main(_):
    wandb.init(project='offroad_states', notes=FLAGS.comment)
    wandb.config.update(FLAGS)

    ## offroad_env stuff
    env = gym.make(FLAGS.env_name)
    env = TimeLimit(env, max_episode_steps=25000)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

    eval_env = gym.make(FLAGS.env_name)
    # We can filter observations directly in the env for evaluation, because we don't need extra information for relabeling.
    eval_env = SelectObs(eval_env, 'states')
    eval_env = TimeLimit(eval_env, max_episode_steps=5000)
    ## offroad_env stuff

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent: SACLearner = globals()[model_cls].create(
        FLAGS.seed, filter_obs_space(env.observation_space), env.action_space,
        **kwargs)

    if FLAGS.expert_replay_buffer:
        with open(FLAGS.expert_replay_buffer, 'rb') as f:
            expert_replay_buffer = pickle.load(f)

    replay_buffer_size = FLAGS.replay_buffer_size
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size)
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={
        'batch_size': FLAGS.batch_size,
        'sample_futures': True,
        'relabel': True,
    })
    if FLAGS.expert_replay_buffer:
        expert_replay_buffer_iterator = expert_replay_buffer.get_iterator(
            sample_args={
                'batch_size': FLAGS.batch_size,
                'sample_futures': True,
                'relabel': True,
            })

    observation, done = env.reset(), False

    def do_relabel_batch(batch):
        nonlocal env
        return filter_batch(
            relabel(batch, env.unwrapped._env._task, from_states=True))

    if FLAGS.expert_replay_buffer:
        expert_replay_buffer._relabel_fn = do_relabel_batch
    replay_buffer._relabel_fn = do_relabel_batch

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(filter_obs(observation))
            action = np.clip(action, env.action_space.low,
                             env.action_space.high)

        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                wandb.log({f'training/{decode[k]}': v}, step=i)

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            agent, update_info = agent.update(batch, utd_ratio=FLAGS.utd_ratio)

            update_info_expert = {}
            if FLAGS.expert_replay_buffer:
                batch_expert = next(expert_replay_buffer_iterator)
                agent, update_info_expert = agent.update(
                    batch_expert,
                    utd_ratio=FLAGS.utd_ratio,
                    update_temperature=False)

            if i % FLAGS.log_interval == 0:
                wandb.log({f'training/running_return': info['running_return']})
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)
                for k, v in update_info_expert.items():
                    wandb.log({f'training/expert/{k}': v}, step=i)

        if i % FLAGS.eval_interval == 0 or i == 100:
            if FLAGS.save_buffer:
                dataset_folder = os.path.join('datasets')
                os.makedirs('datasets', exist_ok=True)
                dataset_file = os.path.join(dataset_folder,
                                            f'{FLAGS.env_name}')
                with open(dataset_file, 'wb') as f:
                    replay_buffer._relabel_fn = None
                    pickle.dump(replay_buffer, f)
                    replay_buffer._relabel_fn = do_relabel_batch

            policy_folder = os.path.join('policies', wandb.run.name)
            os.makedirs(policy_folder, exist_ok=True)
            param_dict = {
                "actor": agent.actor,
                "critic": agent.critic,
                "target_critic": agent.target_critic,
                "temp": agent.temp,
                "rng": agent.rng
            }
            checkpoints.save_checkpoint(policy_folder,
                                        param_dict,
                                        step=i,
                                        keep=1000)

            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)


if __name__ == '__main__':
    app.run(main)
