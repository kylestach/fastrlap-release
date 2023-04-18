import sys
import numpy as np
from offroad_sim_v2.envs import make_dmc_env_record, make_car_task_gym, make_car_task_gym_states
from jaxrl5.data import MemoryEfficientReplayBuffer, ReplayBuffer
from absl import app, flags

import warnings

warnings.filterwarnings("ignore")

env_gym = make_car_task_gym()
observation_space = env_gym.observation_space
rb = MemoryEfficientReplayBuffer(observation_space=observation_space,
                                 action_space=env_gym.action_space,
                                 capacity=30000)
env = make_dmc_env_record(rb)

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_file", None, "Path to the dataset file")

speed = 0.
steer = 0.

obs = None


def manual_policy(time_step):
    del time_step
    global speed, steer
    return [steer, speed]


def forward():
    global speed
    if speed < 0:
        speed = 0
    else:
        speed += 0.3


def brake():
    global speed
    if speed > 0:
        speed = 0
    else:
        speed -= 0.3


def right():
    global steer
    if steer > 0:
        steer = 0
    else:
        steer -= 0.15


def left():
    global steer
    if steer < 0:
        steer = 0
    else:
        steer += 0.15


def main(_):
    import dm_control.viewer.application
    from dm_control.viewer import user_input
    import pickle

    try:
        app = dm_control.viewer.application.Application(title="Offroad",
                                                        width=1024,
                                                        height=768)
    except:
        # Dummy try-except because Application trivially raises an exception in
        # the editor's static analysis, but works fine at runtime.
        pass

    app._input_map.bind(forward, user_input.KEY_UP)
    app._input_map.bind(brake, user_input.KEY_DOWN)
    app._input_map.bind(right, user_input.KEY_RIGHT)
    app._input_map.bind(left, user_input.KEY_LEFT)
    app.launch(environment_loader=env, policy=manual_policy)

    if FLAGS.dataset_file is not None:
        with open(FLAGS.dataset_file, 'wb') as f:
            pickle.dump(rb, f)


if __name__ == '__main__':
    app.run(main)