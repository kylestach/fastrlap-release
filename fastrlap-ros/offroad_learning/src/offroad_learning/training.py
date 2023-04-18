import os
import time
import warnings

warnings.filterwarnings("ignore")

import queue
import multiprocessing
import threading
from tensor_dict_msgs.msg import TensorDict
import tensor_dict_convert
import numpy as np
from jaxrl5.agents import DrQLearner, SACLearner, IQLLearner, PixelIQLLearner
from flax.core import frozen_dict
from jaxrl5.data import MemoryEfficientReplayBuffer, ReplayBuffer
from gym import spaces
import wandb
from absl import app, flags
from ml_collections import config_flags
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import rospkg
import optax
import tqdm
import rospy
import pickle
import json

rospack = rospkg.RosPack()

flags.DEFINE_string("env_name", "offroad_sim/CarTask-v0", "Environment name.")
flags.DEFINE_string("comment", "", "Comment for W&B")
flags.DEFINE_string(
    "expert_replay_buffer", "", "(Optional) Expert replay buffer pickle file."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("eval_interval", 3000, "Number of steps between evaluations.")
flags.DEFINE_integer("log_interval", 100, "Logging interval.")
flags.DEFINE_integer("batch_size", 512, "Mini batch size.")
flags.DEFINE_integer(
    "start_training", int(1024), "Number of training steps to start training."
)
flags.DEFINE_integer("num_stack", 1, "Stack frames.")
flags.DEFINE_integer("replay_buffer_size", 50000, "Capacity of the replay buffer.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
flags.DEFINE_integer("utd_ratio", 8, "Updates per data point")
flags.DEFINE_boolean("use_pixels", True, "Whether to use pixels")
flags.DEFINE_boolean("use_pixel_embeddings", False, "Whether to use pixel embeddings")

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

# Offline config
flags.DEFINE_string("pretrained_checkpoint", "", "Pretrained checkpoint directory.")


def load_pretrained(agent, checkpoint_dir, obs_space, ac_space):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=None)

    actor_params = frozen_dict.unfreeze(agent.actor.params)
    for k, v in loaded_params["actor"]["params"].items():
        if 'encoder' not in k:
            actor_params[k] = v
    critic_params = frozen_dict.unfreeze(agent.critic.params)
    # for k, v in loaded_params["critic"]["params"].items():
    #     if 'encoder' not in k:
    #         critic_params[k] = v

    return agent.replace(
        actor=agent.actor.replace(
            params=frozen_dict.freeze(actor_params),
        ),
        critic=agent.critic.replace(
            params=frozen_dict.freeze(critic_params),
        ),
        target_critic=agent.critic.replace(
            params=frozen_dict.freeze(critic_params),
        ),
    )


def make_action_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(2,))


def filter_obs(obs, use_pixels, use_pixel_embeddings):
    if use_pixels:
        return frozen_dict.freeze(
            {k: obs[k] for k in obs.keys() if k in ["pixels", "states"]}
        )
    elif use_pixel_embeddings:
        return frozen_dict.freeze(
            {k: obs[k] for k in obs.keys() if k in ["image_embeddings", "states"]}
        )
    else:
        return obs["states"]


def filter_batch(obs, use_pixels, use_pixel_embeddings):
    return frozen_dict.freeze(
        {
            "observations": filter_obs(obs["observations"], use_pixels, use_pixel_embeddings),
            "actions": obs["actions"],
            "next_observations": filter_obs(obs["next_observations"], use_pixels, use_pixel_embeddings),
            "rewards": obs["rewards"],
            "dones": obs["dones"],
            "masks": obs["masks"],
        }
    )


def do_relabel_batch(batch):
    # TODO: Perform relabeling
    return batch  # filter_batch(relabel(batch, env.unwrapped._env._task))


def data_to_space(data):
    # Convert a data dict or array into a gym space
    if isinstance(data, dict):
        return spaces.Dict({k: data_to_space(v) for k, v in data.items()})
    elif isinstance(data, np.ndarray):
        return spaces.Box(low=-np.inf, high=np.inf, shape=data.shape)
    else:
        raise ValueError(f"Unknown data type f{type(data)}")


class TrainingRosInterface:
    def __init__(self, rb_queue, param_queue):
        rospy.init_node("training_node")

        self.rb_queue = rb_queue
        self.param_queue = param_queue

        self.param_publisher = rospy.Publisher(
            rospy.get_param("~param_topic", "/actor_params"),
            TensorDict,
            queue_size=1,
            latch=True,
        )
        self.rb_subscribe = rospy.Subscriber(
            rospy.get_param("~rb_topic", "/replay_buffer_data"),
            TensorDict,
            self.rb_callback,
        )

        self.param_publish_callback = rospy.Timer(rospy.Duration(2), self.param_pub_callback)

    def rb_callback(self, msg):
        # Get batches from the queue
        obs = tensor_dict_convert.from_ros_msg(msg)
        self.rb_queue.put(obs)

    def param_pub_callback(self, _):
        params = None
        while True:
            try:
                params = self.param_queue.get_nowait()
            except queue.Empty:
                break
        if params is not None:
            self.param_publisher.publish(
                tensor_dict_convert.to_ros_msg(params)
            )


class TrainingDummyInterface:
    def __init__(self, rb_queue, param_queue):
        self.rb_queue = rb_queue
        self.param_queue = param_queue

        for _ in range(256):
            self.rb_queue.put(
                {
                    "observations": {
                        "pixels": np.random.rand(128, 128, 3, 1),
                        "states": np.random.rand(2),
                        "action": np.random.rand(2),
                    },
                    "actions": np.random.rand(2),
                    "next_observations": {
                        "pixels": np.random.rand(128, 128, 3, 1),
                        "states": np.random.rand(2),
                        "action": np.random.rand(2),
                    },
                    "rewards": np.random.rand(1),
                    "dones": np.array([0]),
                    "masks": np.array([1]),
                }
            )

class Trainer:
    def __init__(
        self,
        args,
        rb_queue,
        param_queue,
    ):
        self.args = args
        self.rb_queue = rb_queue
        self.param_queue = param_queue

        self.agent: DrQLearner = None

        # Replay buffer is none until we get the first data back from the robot
        self.rb = None
        self.rb_iterator = None
        self.expert_rb = None
        self.expert_rb_iterator = None

        self.log_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{args.comment}"

    def update(self):
        update_info = {}
        update_info_expert = {}

        if self.rb is None:
            return {}, {}

        # Train the agent
        if len(self.rb) > self.args.batch_size:
            if self.rb_iterator is None:
                self.rb_iterator = self.rb.get_iterator(
                    queue_size=2,
                    sample_args={
                        "batch_size": self.args.batch_size,
                        "sample_futures": None,
                        "relabel": False,
                    }
                )

            batch = next(self.rb_iterator)
            daction_max = jnp.array([0.2, 0.2])
            output_low = jnp.clip(batch["observations"]["action"] - daction_max, -1, 1)
            output_high = jnp.clip(batch["observations"]["action"] + daction_max, -1, 1)
            batch = filter_batch(batch, self.args.use_pixels, self.args.use_pixel_embeddings)
            self.agent, update_info = self.agent.update(batch, self.args.utd_ratio, output_range=(output_low, output_high))
        if self.expert_rb and len(self.expert_rb) > self.args.batch_size:
            batch_expert = next(self.expert_rb_iterator)
            daction_max = jnp.array([0.2, 0.2])
            output_low = jnp.clip(batch_expert["observations"]["action"] - daction_max, -1, 1)
            output_high = jnp.clip(batch_expert["observations"]["action"] + daction_max, -1, 1)
            batch_expert = filter_batch(batch_expert, self.args.use_pixels, self.args.use_pixel_embeddings)
            self.agent, update_info_expert = self.agent.update(batch_expert, self.args.utd_ratio, output_range=(output_low, output_high))

        return update_info, update_info_expert

    def main(self):
        config = None
        if self.args.use_pixels:
            config = self.args.config_pixels
        elif self.args.use_pixel_embeddings:
            config = self.args.config_embeddings
        else:
            config = self.args.config_states
        config = dict(config)

        comment = self.args.comment
        if self.args.pretrained_checkpoint:
            comment += f" (pretrained from {self.args.pretrained_checkpoint})"
        if self.args.expert_replay_buffer:
            comment += f" (with expert data {self.args.expert_replay_buffer})"

        wandb.init(project="offroad_ros_gazebo", config=config, notes=comment)

        pbar = tqdm.tqdm(total=1000000)

        i = 0
        num_env_steps = 0

        vehicle_metrics = {}
        summary = {
            "lap_times": [],
            "num_recovery": 0,
        }

        while True:
            while True:
                try:
                    batch = self.rb_queue.get_nowait()

                    if batch == 'shutdown':
                        return

                    num_env_steps += batch["rewards"].shape[0]

                    if self.agent is None:
                        use_pixels = self.args.use_pixels
                        use_pixel_embeddings = self.args.use_pixel_embeddings
                        kwargs = config
                        model_cls = kwargs.pop("model_cls")
                        action_space = make_action_space()

                        if use_pixels:
                            # For pixels/states, we want a dict space.
                            filtered_obs_space = data_to_space({
                                'pixels': batch["observations"]["pixels"],
                                'states': batch["observations"]["states"],
                            })
                        elif use_pixel_embeddings:
                            # For pixels/states, we want a dict space.
                            filtered_obs_space = data_to_space({
                                'image_embeddings': batch["observations"]["image_embeddings"],
                                'states': batch["observations"]["states"],
                            })
                        else:
                            # For states only, we just want an array.
                            filtered_obs_space = data_to_space(batch["observations"]["states"])

                        if use_pixel_embeddings:
                            self.agent = globals()[model_cls].create(
                                self.args.seed, filtered_obs_space, action_space, pixel_embeddings_key="image_embeddings", **kwargs
                            )
                        else:
                            self.agent = globals()[model_cls].create(
                                self.args.seed, filtered_obs_space, action_space, **kwargs
                            )

                        # Load the pretrained model if specified
                        if self.args.pretrained_checkpoint:
                            self.agent = load_pretrained(
                                agent=self.agent,
                                checkpoint_dir=self.args.pretrained_checkpoint,
                                obs_space=filtered_obs_space,
                                ac_space=action_space
                            )
                            self.param_queue.put(
                                jax.tree_map(
                                    lambda x: np.asarray(x),
                                    frozen_dict.unfreeze(self.agent.actor.params),
                                )
                            )

                    if self.rb is None:
                        # Build the replay buffer based on the first datapoint
                        if use_pixels:
                            self.rb = MemoryEfficientReplayBuffer(
                                observation_space=data_to_space(batch["observations"]),
                                action_space=action_space,
                                capacity=self.args.replay_buffer_size,
                                pixel_keys=("pixels",) if self.args.use_pixels else (),
                                relabel_fn=do_relabel_batch,
                            )
                        else:
                            self.rb = ReplayBuffer(
                                observation_space=data_to_space(batch["observations"]),
                                action_space=action_space,
                                capacity=self.args.replay_buffer_size,
                                # pixel_keys=("pixels",) if self.args.use_pixels else (),
                                relabel_fn=do_relabel_batch,
                            )

                    if self.expert_rb is None and self.args.expert_replay_buffer:
                        self.expert_rb = pickle.load(open(self.args.expert_replay_buffer, "rb"))
                        self.expert_rb_iterator = self.expert_rb.get_iterator(
                            sample_args={
                                "batch_size": self.args.batch_size,
                                "sample_futures": None,
                                "relabel": False,
                            }
                        )

                    vehicle_metrics.update(batch.pop("wandb_logging"))
                    self.rb.insert(batch)
                except queue.Empty:
                    break

            update_info, update_info_expert = self.update()
            if update_info != {} or update_info_expert != {}:
                pbar.update(1)

            if i >= self.args.start_training:
                self.agent = self.agent.replace(target_entropy=self.agent.target_entropy - 4e-5)
                if i % self.args.log_interval == 0:
                    wandb_log = {
                        f"training/target_entropy": np.array(self.agent.target_entropy) if self.agent else 0,
                    }
                    for k, v in update_info.items():
                        wandb_log.update({f"training/{k}": np.array(v)})
                    for k, v in update_info_expert.items():
                        wandb_log.update({f"training/expert/{k}": np.array(v)})
                    wandb_log["environment steps"] = num_env_steps

                    for k, v in vehicle_metrics.items():
                        wandb_log.update({f"environment/{k}": np.array(v) if v.shape == (1,) else wandb.Histogram(v)})
                    
                    os.makedirs(os.path.join("summaries", wandb.run.name), exist_ok=True)
                    if 'num_recovery' in vehicle_metrics:
                        summary['num_recovery'] = vehicle_metrics['num_recovery'].item()
                    if 'lap_times' in vehicle_metrics:
                        summary['lap_times'] = vehicle_metrics['lap_times'].tolist()
                    json.dump(summary, open(os.path.join("summaries", wandb.run.name, "summary.json"), "w"))

                    vehicle_metrics = {}

                    wandb.log(wandb_log, step=i)

            if i % 20 == 0 and self.agent is not None:
                self.param_queue.put(
                    jax.tree_map(
                        lambda x: np.asarray(x),
                        frozen_dict.unfreeze(self.agent.actor.params),
                    )
                )

            if (i + 1) % self.args.eval_interval == 0:
                dataset_folder = os.path.join('datasets', wandb.run.name)
                policy_folder = os.path.join('policies', wandb.run.name)
                os.makedirs(policy_folder, exist_ok=True)
                os.makedirs(dataset_folder, exist_ok=True)

                if self.args.save_buffer:
                    print("Saving dataset to: " + dataset_folder)
                #     self.rb.save(dataset_folder, i)
                    pickle.dump(self.rb, open(os.path.join(dataset_folder, f"dataset_{i}.pkl"), "wb"))
                    print("Saved dataset: " + dataset_folder)

                param_dict = {
                    "actor": self.agent.actor,
                    "critic": self.agent.critic,
                    "target_critic_params": self.agent.target_critic,
                    "temp": self.agent.temp,
                    "rng": self.agent.rng
                }
                checkpoints.save_checkpoint(policy_folder,
                                            param_dict,
                                            step=i,
                                            keep=1000)

            if num_env_steps > self.args.batch_size:
                i += 1


def main(_):
    args = flags.FLAGS

    rb_queue = queue.Queue()
    param_queue = queue.Queue()

    trainer = Trainer(args, rb_queue, param_queue)

    process = threading.Thread(target=trainer.main)
    process.start()

    ros_iface = TrainingRosInterface(rb_queue, param_queue)
    # ros_iface = TrainingDummyInterface(rb_queue, param_queue)
    rospy.spin()
    rb_queue.put('shutdown')

    process.join()


if __name__ == "__main__":
    app.run(main)
