from typing import Union
import gym
from gym import spaces
import numpy as np
from offroad_sim_v2.envs import make_car_task_gym, make_car_task_gym_states
from jaxrl5.agents import SACLearner, DrQLearner
import jax
from ml_collections import config_flags
from flax.training.checkpoints import restore_checkpoint
from dmcgym.env import DMCGYM
from dm_control.composer import Environment as DMCEnvironment
from tqdm import trange
from pathlib import Path
from PIL import Image, ImageDraw
import os
from flax.core.frozen_dict import FrozenDict
from moviepy.editor import ImageSequenceClip

from absl import app, flags

jax.config.update('jax_platform_name', 'cpu')

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_file", None, "Path to the policy file")
flags.DEFINE_string('goal_config', 'small_inner_graph',
                    'Goal configuration name')
flags.DEFINE_string("video_output_dir", None,
                    "Path to the video output directory")
flags.DEFINE_integer("num_trajectories", 1, "Number of trajectories to run")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_boolean(
    'pixels', True,
    'True if the agent is from pixels or False for from states')
config_flags.DEFINE_config_file(
    'config_pixels',
    'configs/drq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)
config_flags.DEFINE_config_file(
    'config_states',
    'configs/redq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


class FilterObs(gym.ObservationWrapper):
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys
        self.observation_space = spaces.Dict(
            {k: env.observation_space[k]
             for k in keys})

    def observation(self, obs):
        return {k: obs[k] for k in self.keys}


class SelectObs(gym.ObservationWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self.key = key
        self.observation_space = env.observation_space[self.key]

    def observation(self, obs):
        return obs[self.key]


import warnings

warnings.filterwarnings("ignore")


def recursive_shape(x):
    if isinstance(x, (np.ndarray, jax.numpy.ndarray)):
        return x.shape
    if isinstance(x, (dict, FrozenDict)):
        return {k: recursive_shape(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(recursive_shape(v) for v in x)
    print(type(x))
    return None


# Load the policy from a file
def load_agent(agent: Union[DrQLearner, SACLearner], policy_file: str):
    target_critic_key = "target_critic_params" if FLAGS.pixels else "target_critic"
    param_dict = {
        "actor": agent.actor,
        "critic": agent.critic,
        target_critic_key: agent.target_critic,
        "temp": agent.temp,
        "rng": agent.rng
    }
    param_dict = restore_checkpoint(policy_file, param_dict)
    return agent.replace(actor=param_dict["actor"],
                         critic=param_dict["critic"],
                         target_critic=param_dict[target_critic_key],
                         temp=param_dict["temp"],
                         rng=param_dict["rng"])


def run_trajectory(agent, env: gym.Env, max_steps=5000, render_fn=None):
    obs = env.reset()
    images = []
    for _ in trange(max_steps):
        action, agent = agent.sample_actions(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, info = env.step(action)

        if render_fn is not None:
            image = Image.fromarray(render_fn(env))
            draw = ImageDraw.Draw(image)
            state = obs['states'] if FLAGS.pixels else obs
            # draw.text((10, 10), f"Speed: {state[6]:.2f}", fill=(255, 255, 255))
            draw.text((10, 10),
                      f"Goal: {state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}",
                      fill=(255, 255, 255))
            images.append(np.asarray(image))

        if done:
            break

    return images


def render_fn(env: DMCGYM):
    return env.render(camera_id='car/overhead_track', width=480, height=480)


def main(_):
    if FLAGS.pixels:
        env_gym = make_car_task_gym(goal_config=FLAGS.goal_config)
        env_gym = FilterObs(env_gym, ["states", "pixels"])
        kwargs = dict(FLAGS.config_pixels)
    else:
        env_gym = make_car_task_gym_states(goal_config=FLAGS.goal_config)
        env_gym = SelectObs(env_gym, 'states')
        kwargs = dict(FLAGS.config_states)

    model_cls = kwargs.pop("model_cls")
    agent: SACLearner = globals()[model_cls].create(FLAGS.seed,
                                                    env_gym.observation_space,
                                                    env_gym.action_space,
                                                    **kwargs)

    agent = load_agent(agent, FLAGS.policy_file)
    Path(FLAGS.video_output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(FLAGS.num_trajectories):
        images = run_trajectory(agent, env_gym, render_fn=render_fn)
        ImageSequenceClip(sequence=images, fps=20).write_videofile(
            os.path.join(
                FLAGS.video_output_dir,
                f"video-{os.path.basename(FLAGS.policy_file)}-{i}.mp4"))


if __name__ == '__main__':
    app.run(main)
