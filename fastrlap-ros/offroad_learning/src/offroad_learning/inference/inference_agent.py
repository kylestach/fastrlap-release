from typing import Union, Type, Any
import rospy
import time
import collections
from functools import partial

import numpy as np
import json
from datetime import datetime

from ackermann_msgs.msg import AckermannDriveStamped
import gazebo_msgs.msg as gzm
import gazebo_msgs.srv as gzs
import geometry_msgs.msg as gm
import std_msgs.msg as stdm
import tensor_dict_msgs.msg as tm

from gym import spaces
from flax.core import frozen_dict
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax

import tensor_dict_convert
from .aggregator import RosDataAggregator
from .state_machine import InferenceStateMachine, State
from .goal_graph import GoalGraph


from jaxrl5.networks.encoders import D4PGEncoder


class StackedImageNetEncoder(nn.Module):
    encoder: Type[nn.Module] = partial(D4PGEncoder, (32, 32, 32, 32), (3, 3, 3, 3), (2, 2, 2, 2), padding='VALID')

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (*x.shape[:-1], 3, 3))
        StackedEncoder = nn.vmap(self.encoder, in_axes=-1, out_axes=-1, variable_axes={'params': None}, split_rngs={'params': False})
        x = StackedEncoder()(x)
        return x.sum(-1)

def load_encoder(pretrained_model):
    encoder_cls = partial(D4PGEncoder, (32, 32, 32, 32), (3, 3, 3, 3), (2, 2, 2, 2))
    encoder = None
    params = checkpoints.restore_checkpoint(pretrained_model, target=None)
    if 'critic' in params:
        encoder_params = params['critic']['params']['encoder_0']
        encoder = encoder_cls()
    elif 'encoder' in params:
        encoder_params = params['encoder']['params']['encoder_0']
        encoder = encoder_cls()
    elif 'params' in params:
        if 'D4PGEncoder_0' in params['params']:
            encoder = StackedImageNetEncoder(encoder=encoder_cls)

            encoder_params = frozen_dict.unfreeze(encoder.init(jax.random.PRNGKey(0), jnp.zeros((1, 128, 128, 9)))['params'])
            encoder_params['VMapD4PGEncoder'] = params['params']['D4PGEncoder_0']
        elif 'Encoder_0' in params['params']:
            encoder_params = params['params']['Encoder_0']
            encoder = encoder_cls()
        else:
            raise ValueError(f"Could not find encoder in pretrained model.params: {jax.tree_map(lambda x: jnp.shape(x), params['params'])}")
    else:
        raise ValueError("Could not find encoder in pretrained model")

    train_state = TrainState.create(
        apply_fn=encoder.apply,
        params=frozen_dict.freeze(encoder_params),
        tx=optax.GradientTransformation(lambda _: None, lambda _: None)
    )
    return train_state


def calculate_next_relative_goal_continuous(goal_odom_t0, pose_2d_odom_t1):
    # Transform matrix from odom to base_link at t=1
    base_link_to_odom_t1 = np.array([
        [np.cos(pose_2d_odom_t1[2]), -np.sin(pose_2d_odom_t1[2]), pose_2d_odom_t1[0]],
        [np.sin(pose_2d_odom_t1[2]), np.cos(pose_2d_odom_t1[2]), pose_2d_odom_t1[1]],
        [0, 0, 1]
    ])
    odom_to_base_link_t1 = np.linalg.inv(base_link_to_odom_t1)

    goal_relative = np.matmul(odom_to_base_link_t1, np.array([goal_odom_t0[0], goal_odom_t0[1], 1]))
    return goal_relative[:2]


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


def obs_to_space(obs):
    if isinstance(obs, (dict, frozen_dict.FrozenDict)):
        return spaces.Dict({k: obs_to_space(v) for k, v in obs.items()})
    elif isinstance(obs, (np.ndarray, jnp.ndarray)):
        return spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape)

    raise ValueError(f"Unknown data type {type(obs)}")


class RosAgent:
    """
    An agent that accepts and aggregates ROS messages from several different sources, and periodically performs inference and publishes a new command.
    """

    def __init__(self, agent_cls, use_pixels, use_pixel_embeddings, sim, num_stack):
        # Create a ROS node for the training and inference process
        self.agent_cls = agent_cls
        self.agent = None
        self.use_pixels = use_pixels
        self.use_pixel_embeddings = use_pixel_embeddings
        self.sim = sim
        self.num_stack = num_stack

        self.param_subscription = rospy.Subscriber(
            rospy.get_param("~param_topic", "/actor_params"),
            tm.TensorDict,
            self.receive_params,
        )
        self.rb_publisher = rospy.Publisher(
            rospy.get_param("~rb_topic", "/replay_buffer_data"),
            tm.TensorDict,
            queue_size=100,
        )

        self.last_speeds = collections.deque(maxlen=100)

        if self.use_pixel_embeddings:
            encoder_checkpoint = rospy.get_param("~encoder_checkpoint", None)
            if encoder_checkpoint:
                self.encoder = load_encoder(encoder_checkpoint)
        else:
            self.encoder = None

        self.state_keys = [
            "accel",
            "relative_linear_velocity",
            "relative_angular_velocity",
            "goal_relative",
            "action",
        ] + (["pose_2d"] if not (self.use_pixels or self.use_pixel_embeddings) else [])
        self.ackermann = rospy.get_param("~ackermann", False)

        if self.ackermann:
            self.action_scale = np.array([2.0, 0.5])
            self.action_bias = np.array([2.5, 0.0])
        else:
            self.action_scale = np.array([1.5, 1.0])
            self.action_bias = np.array([0.5, 0.0])

        self.current_lap_collisions = 0
        self.lap_collisions = []

        self.fixed_frame_id = rospy.get_param("~fixed_frame", "map")

        self.state_machine = InferenceStateMachine(self.state_keys, self.action_scale, self.action_bias)
        self.goal_graph = GoalGraph(self.fixed_frame_id)
        self.aggregator = RosDataAggregator(self.num_stack, self.control_callback, self.state_keys, self.fixed_frame_id, encoder=self.encoder)

        self.i = 0

        if self.ackermann:
            self.action_publisher = rospy.Publisher(
                rospy.get_param("~action_topic", "/command"),
                AckermannDriveStamped,
                queue_size=1,
            )
        else:
            self.action_publisher = rospy.Publisher(
                rospy.get_param("~action_topic", "/command"),
                gm.Twist,
                queue_size=1,
            )

        if self.sim:
            self.state_reset = rospy.ServiceProxy(
                "/gazebo/set_model_state", gzs.SetModelState
            )

        self.mode_publisher = rospy.Publisher(
            rospy.get_param("~mode_topic", "/offroad_learning/mode"), stdm.String, queue_size=1
        )

        if self.ackermann:
            self.teleop_record_subscriber = rospy.Subscriber(
                rospy.get_param("~teleop_record_command", "/offroad_learning/teleop_record_command"),
                AckermannDriveStamped,
                self.receive_teleop_record,
            )
            self.teleop_subscriber = rospy.Subscriber(
                rospy.get_param("~teleop_command", "/joy_teleop/cmd_vel"),
                AckermannDriveStamped,
                self.receive_teleop,
            )
        else:
            self.teleop_record_subscriber = rospy.Subscriber(
                rospy.get_param("~teleop_record_command", "/offroad_learning/teleop_record_command"),
                gm.Twist,
                self.receive_teleop_record,
            )
            self.teleop_subscriber = rospy.Subscriber(
                rospy.get_param("~teleop_command", "/joy_teleop/cmd_vel"),
                gm.Twist,
                self.receive_teleop,
            )

        self.last_teleop_time = rospy.Time.now()

        self.last_actions = collections.deque(maxlen=30)

    def receive_teleop_record(self, msg: Union[AckermannDriveStamped, gm.Twist]):
        if self.ackermann:
            self.teleop_command = np.array([msg.drive.speed, msg.drive.steering_angle])
        else:
            self.teleop_command = np.array([msg.linear.x, msg.angular.z])
        self.teleop_command = (self.teleop_command - self.action_bias) / self.action_scale
        self.state_machine.handle_teleop_record()

    def obs_full(self, obs):
        return all(k in obs for k in self.state_keys)

    def receive_teleop(self, msg):
        self.state_machine.handle_teleop()

    def receive_params(self, msg: tm.TensorDict):
        if self.agent is not None:
            params = tensor_dict_convert.from_ros_msg(msg)
            new_actor = self.agent.actor.replace(params=params)
            self.agent = self.agent.replace(actor=new_actor)

    def make_agent(self, obs):
        obs_space = obs_to_space(obs)
        return self.agent_cls(observation_space=obs_space)

    def control_callback(self, obs):
        old_state = self.state_machine.state
        should_record, truncated, terminated = self.state_machine.tick_state(obs)
        state = self.state_machine.state

        lap_time = None

        if "position" in obs:
            reached_goal, finished_lap, lap_time = self.goal_graph.tick(obs["position"])
            if state != old_state and old_state == State.RECOVERY:
                self.current_lap_collisions += 1
            if finished_lap:
                self.lap_collisions.append(self.current_lap_collisions)
                self.current_lap_collisions = 0
                lap_data = {
                    'lap_times': list(self.goal_graph.lap_times),
                    'lap_collisions': list(self.lap_collisions),
                }
                now = datetime.now()
                with open(f'lap-times-{now.strftime("%Y-%m-%d_%H.%M.%S")}.txt', "w") as f:
                    json.dump(lap_data, f)
            self.aggregator.receive_goal(self.goal_graph.goal_poses())

        if should_record and self.obs_full(self.aggregator.prev_observation) and self.obs_full(obs):
            did_fail = self.state_machine.state == State.CRASHED or self.state_machine.state == State.RECOVERY
            if did_fail:
                rospy.logwarn(f"Failed with max accel: {obs['max_accel_hist']}")
            fail_cost = (10 + np.clip(0, obs["max_accel_hist"] / 5, 5)) * did_fail
            rewards = np.sum(obs["relative_linear_velocity"][:2] * obs["goal_relative"][:2]) - fail_cost # + 30 * reached_goal

            # Calculate goal and next goal in previous obs's odom frame

            observations = self.aggregator.prev_observation.copy()
            actions = self.aggregator.prev_action.copy()
            next_observations = obs.copy()

            # next_goal_relative = calculate_next_relative_goal_continuous(next_observations["goal_odom"], next_observations["pose_2d_odom"])
            # next_goal_relative_distance = np.linalg.norm(next_goal_relative)
            # next_observations["goal_relative"] = np.array([
            #     next_goal_relative[0] / next_goal_relative_distance,
            #     next_goal_relative[1] / next_goal_relative_distance,
            #     next_goal_relative_distance
            # ])

            self.last_speeds.append(np.linalg.norm(observations["relative_linear_velocity"][:2]))
            wandb_logs = {
                "num_recovery": np.array([self.state_machine.num_recovery]),
            }
            if lap_time is not None:
                wandb_logs["lap_time"] = np.array([lap_time])
                if self.goal_graph.lap_times:
                    wandb_logs["lap_times"] = np.asarray(self.goal_graph.lap_times)
                    wandb_logs["best_lap_time"] = np.min(np.asarray(self.goal_graph.lap_times), keepdims=True)
            if len(self.last_speeds) > 1:
                wandb_logs["average_speed"] = np.mean(np.asarray(self.last_speeds), keepdims=True)

            batch = {
                "observations": observations,
                "actions": actions,
                "next_observations": next_observations,
                "rewards": np.array([rewards]),
                "dones": np.array([1.0 if (terminated or truncated) else 0.0]),
                "masks": np.array([0.0 if terminated else 1.0]),
                "wandb_logging": wandb_logs,
            }
            self.rb_publisher.publish(tensor_dict_convert.to_ros_msg(batch))

        action = np.zeros(2)
        if self.state_machine.state == State.TELEOP_RECORD:
            action = np.clip(self.teleop_command, -1, 1)
            self.publish_action(action)
            # rospy.logwarn(f' action Teleop learning:  {action}')
        elif self.state_machine.state == State.LEARNING:
            prev_action = self.aggregator.prev_action
            daction_max = jnp.array([0.2, 0.2])
            low = jnp.clip(prev_action - daction_max, -1.0, 1.0)
            high = jnp.clip(prev_action + daction_max, -1.0, 1.0)

            obs_filtered = filter_obs(obs, self.use_pixels, self.use_pixel_embeddings)

            if self.agent is None:
                self.agent = self.make_agent(obs_filtered)

            action, self.agent = self.agent.sample_actions(obs_filtered, output_range=(low, high))

            #Bandaid
            if np.any(np.isnan(action)):
                print(f"Got NaN in action. NaN in obs: {jax.tree_map(lambda x: np.any(np.isnan(x)), obs_filtered)}")
                action = np.zeros(2)

            action = np.clip(action, -1.0, 1.0)
            self.publish_action(action)
            self.last_actions.append(action[0])
        elif self.state_machine.state == State.CRASHED:
            # rospy.logwarn(f' action crashed:  {action}')
            self.uninvert_sim()
        elif self.state_machine.state == State.RECOVERY:
            self.aggregator.accel_hist = collections.deque(maxlen=50)
            action = np.array([-1.0, self.state_machine.recovery_steer * self.action_scale[1]])
            self.publish_action(action, raw=True)
            # rospy.logwarn(f' action rec:  {action}')

        # rospy.logwarn(f' action:  {action}')
        return action
    
    def publish_action(self, action, raw=False):
        if raw:
            action_scaled = action
        else:
            action_scaled = action * self.action_scale + self.action_bias
        if self.ackermann:
            msg = AckermannDriveStamped()
            msg.drive.speed = action_scaled[0]
            msg.drive.steering_angle = action_scaled[1]
        else:
            msg = gm.Twist()
            msg.linear.x = action_scaled[0]
            msg.angular.z = action_scaled[1]
        self.action_publisher.publish(msg)

    def uninvert_sim(self):
        if self.state_machine.state == State.CRASHED and self.sim:
            self.state_reset(gzs.SetModelStateRequest(gzm.ModelState(
                    model_name='jackal',
                    pose=gm.Pose(
                        position=gm.Point(35, -13, 1),
                        orientation=gm.Quaternion(0, 0, 1, 0)
                    ),
                    reference_frame=self.fixed_frame_id,
            )))
