import collections
from dm_control import composer
from dm_control.composer.observation import observable as observable_lib
import numpy as np
from dm_control.utils import transformations
from dm_control.mjcf.physics import Physics
import quaternion as npq

from .car import Car
from .heightfield_arena import HeightFieldArena
from . import utils as env_helpers
from .goal_graph import GoalGraph

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.005


class CarTask(composer.Task):

    def __init__(self,
                 step_timeout: int = 5000,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
                 scale: float = 20.0,
                 include_camera: bool = True,
                 goal_config='small_inner_graph'):
        self._arena = HeightFieldArena(scale=scale)

        self._car = Car()
        self._arena.add_free_entity(self._car)

        self._car.observables.enable_all()
        if not include_camera:
            self._car.observables.get_observable('realsense_camera').enabled = False

        self.goal_graph = GoalGraph(scale=scale, goal_config=goal_config)
        self.goal_graph.add_renderables(self._arena._mjcf_root, self._arena.height_lookup)

        self.set_timesteps(control_timestep, physics_timestep)

        self._last_positions = np.empty((500, 2), dtype=np.float32)
        self._last_positions.fill(np.inf)

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        def goal_polar(physics: Physics):
            car_pose_2d = self._car.observables.get_observable('body_pose_2d')(physics)
            return env_helpers.relative_goal_polarcoord(car_pose_2d, self.goal_graph.current_goal)

        relative_polar_coords = observable_lib.Generic(goal_polar)
        relative_polar_coords.enabled = True

        absolute_polar_coords = observable_lib.Generic(lambda physics: self.goal_graph.current_goal)
        absolute_polar_coords.enabled = True

        should_timeout = observable_lib.Generic(self.should_timeout)
        should_timeout.enabled = True

        task_obs = collections.OrderedDict({
            'goal_relative': relative_polar_coords,
            'goal_absolute': absolute_polar_coords,
            'timeout': should_timeout,
        })

        return task_obs

    def initialize_episode(self, physics: Physics, random_state: np.random.RandomState):
        super().initialize_episode(physics, random_state)
        self._arena.initialize_episode(physics, random_state)

        # Reset to pick a new goal
        self.goal_graph.reset(physics)

        start_pos, start_heading_bounds = self.goal_graph.current_start
        start_pos = np.concatenate([start_pos + np.random.uniform(-0.2, 0.2, size=(2,)), [0.5]])
        start_quat = transformations.euler_to_quat([0, 0, np.random.uniform(*start_heading_bounds)])
        self._car.set_pose(physics, start_pos, start_quat)

        car_geoms = [geom for geom in self._car.mjcf_model.find_all('geom')]
        obstacle_geoms = [wall for wall in self._arena.walls]

        # Keep track of geometry for collision detection.
        self._car_geomids = set(physics.bind(car_geoms).element_id)
        self._obstacle_geomids = set(physics.bind(obstacle_geoms).element_id)

        # for geom in self._arena.mjcf_model.find_all('geom'):
        #     geom.friction = (0.0, 0.005, 0.0001)
        #     geom.solref = (2 * DEFAULT_PHYSICS_TIMESTEP, 1.0)
        #     # geom.solimp = (0.95, 0.99, 0.001)

        self._last_positions.fill(np.inf)

    def should_terminate_episode(self, physics: Physics):
        car_pos, car_quat = self._car.get_pose(physics)

        if env_helpers.is_upside_down(car_quat):
            return True

        if self.should_timeout(physics):
            return True

        if not self._arena.in_bounds(car_pos[:2]):
            return True

        return False

    def after_step(self, physics: Physics, random_state: np.random.RandomState):
        car_pos, car_quat = self._car.get_pose(physics)
        self.goal_graph.tick(car_pos, physics)

        self._last_positions = np.roll(self._last_positions, -1, axis=0)
        self._last_positions[-1] = car_pos[:2]

    def get_reward(self, physics: Physics):
        car_pos, car_quat = self._car.get_pose(physics)
        car_pose_2d = self._car.observables.get_observable('body_pose_2d')(physics)
        car_vel_2d = self._car.observables.get_observable('body_vel_2d')(physics)
        goal_absolute = self.goal_graph.current_goal
        should_timeout = self.should_timeout(physics)

        reward = self.batch_compute_reward(car_quat[None], car_pose_2d[None], car_vel_2d[None], goal_absolute[None], should_timeout[None])[0]

        return reward

    def batch_compute_reward(self, car_quat, car_pose_2d, car_vel_2d, goal_absolute, should_timeout):
        # If it's upside down, terminate with negative reward
        upside_down = env_helpers.batch_is_upside_down(car_quat)
        distances_to_goal = np.linalg.norm(car_pose_2d[:, :2] - goal_absolute, axis=-1)

        directions_to_goal = goal_absolute - car_pose_2d[:, :2]
        directions_to_goal /= np.linalg.norm(directions_to_goal, axis=-1, keepdims=True) + 1e-6

        velocities_to_goal = np.sum(car_vel_2d * directions_to_goal, axis=-1)

        r = velocities_to_goal - 100 * upside_down + 100 * (distances_to_goal < self.goal_graph.goal_threshold) - 100 * should_timeout

        return r

    def batch_compute_reward_from_observation(self, observation, action, next_observation):
        # For some reason dm_control computes the reward after the step, so we need to do the same
        car_quat = next_observation['car/body_rotation']
        car_pose_2d = next_observation['car/body_pose_2d']
        car_vel_2d = next_observation['car/body_vel_2d']
        goal_absolute = next_observation['goal_absolute']
        should_timeout = next_observation['timeout']

        return self.batch_compute_reward(car_quat, car_pose_2d, car_vel_2d, goal_absolute, should_timeout)

    def sample_goals_future(self, observations, next_observations, future_observations):
        # Sample goals relative to the current pose
        car_pose_t0 = observations['car/body_pose_2d']
        car_pose_t1 = next_observations['car/body_pose_2d']

        goal_absolute = future_observations['car/body_pose_2d'][:, :2]

        goal_polar_t0 = env_helpers.batch_relative_goal_polarcoord(car_pose_t0, goal_absolute)
        goal_polar_t1 = env_helpers.batch_relative_goal_polarcoord(car_pose_t1, goal_absolute)

        return ((goal_absolute, goal_polar_t0), (goal_absolute, goal_polar_t1))

    def sample_goals_random(self, batch_size, observations, next_observations):
        # Sample goals relative to the current pose
        car_pose_t0 = observations['car/body_pose_2d']
        car_pos_t0 = observations['car/body_pose_2d'][:, :2]
        car_yaw_t0 = observations['car/body_pose_2d'][:, 2]
        car_pose_t1 = next_observations['car/body_pose_2d']

        theta = np.random.normal(0, 1.0, size=(batch_size))
        distance = np.random.normal(2.5, 1.5, size=(batch_size))

        # The car to goal vector, in world coordinates, at t=0
        car_to_goal_world_t0 = np.stack([np.cos(theta+car_yaw_t0) * distance, np.sin(theta+car_yaw_t0) * distance], axis=-1)

        # The absolute goal position, in world coordinates, at t=0 and t=1
        goal_absolute = car_pos_t0 + car_to_goal_world_t0

        goal_polar_t0 = env_helpers.batch_relative_goal_polarcoord(car_pose_t0, goal_absolute)
        goal_polar_t1 = env_helpers.batch_relative_goal_polarcoord(car_pose_t1, goal_absolute)

        return ((goal_absolute, goal_polar_t0), (goal_absolute, goal_polar_t1))

    def should_timeout(self, physics: Physics):
        car_pos, car_quat = self._car.get_pose(physics)
        return np.array(np.linalg.norm(car_pos[None, :2] - self._last_positions).max() < 1.0, dtype=np.float32)