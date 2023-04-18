from functools import partial
from gym import spaces
import numpy as np
import rospy

from jaxrl5.agents.drq.drq_learner import DrQLearner, SACLearner

from .inference_agent import RosAgent

from absl import app, flags
from ml_collections import config_flags

import rospkg
rospack = rospkg.RosPack()

flags.DEFINE_integer("seed", 0, "Random seed.")
config_flags.DEFINE_config_file(
    "config_pixels",
    f"{rospack.get_path('offroad_learning')}/config/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_embeddings",
    f"{rospack.get_path('offroad_learning')}/config/redq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_states",
    f"{rospack.get_path('offroad_learning')}/config/redq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def make_action_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(2,))

def main(_):
    rospy.init_node("inference")

    pixels = rospy.get_param("~use_pixels", False)
    pixel_embeddings = rospy.get_param("~use_pixel_embeddings", False)
    sim = rospy.get_param("~sim", True)
    num_stack = rospy.get_param("~num_stack", 3)

    assert not (pixels and pixel_embeddings), "Can't use both pixels and pixel embeddings"

    args = flags.FLAGS

    if pixels:
        config = args.config_pixels
    elif pixel_embeddings:
        config = args.config_embeddings
    else:
        config = args.config_states
    kwargs = dict(config)
    model_cls = kwargs.pop("model_cls")
    if pixel_embeddings:
        agent_cls = partial(globals()[model_cls].create, seed=args.seed, action_space=make_action_space(), pixel_embeddings_key="image_embeddings", **kwargs)
    else:
        agent_cls = partial(globals()[model_cls].create, seed=args.seed, action_space=make_action_space(), **kwargs)
    agent = RosAgent(agent_cls, pixels, pixel_embeddings, sim, num_stack)
    rospy.spin()
