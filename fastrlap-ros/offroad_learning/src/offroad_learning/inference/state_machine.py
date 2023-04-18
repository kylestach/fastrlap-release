from enum import Enum
import collections

import rospy
import std_msgs.msg as stdm

import numpy as np

# Enum for the state machine
class State(Enum):
    STARTING = 0
    LEARNING = 1
    CRASHED = 2
    ERRORED = 3
    TELEOP = 4
    TELEOP_RECORD = 5
    RECOVERY = 6


class InferenceStateMachine:
    def __init__(self, state_keys, action_scale, action_bias):
        self.state = State.STARTING
        self.state_keys = state_keys
        self.last_teleop_time = rospy.Time.now()
        self.start_recovery_time = rospy.Time.now()
        self.positions = collections.deque(maxlen=30)
        self.should_record_last = False
        self.mode_publisher = rospy.Publisher(
            rospy.get_param("~mode_topic", "/offroad_learning/mode"), stdm.String, queue_size=1
        )
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.recovery_forwards = False

        self.num_recovery = 0

    def obs_full(self, obs):
        print({k: k in obs for k in self.state_keys})
        return all(k in obs for k in self.state_keys)

    def is_stuck(self, obs):
        return len(self.positions) == self.positions.maxlen and np.all(np.linalg.norm(np.stack(self.positions) - obs["position"], axis=-1) < 0.3)

    def is_inverted(self, obs):
        return obs["accel"][2] < -4.0 or obs["max_accel_hist"] > 20

    def should_record(self, obs):
        return self.obs_full(obs) and (self.state == State.LEARNING or self.state == State.TELEOP_RECORD)

    def handle_teleop(self):
        self.state = State.TELEOP
        self.last_teleop_time = rospy.Time.now()

    def handle_teleop_record(self):
        self.state = State.TELEOP_RECORD
        self.last_teleop_time = rospy.Time.now()

    def tick_state(self, obs):
        # Log the current position for the stuck calculator
        if 'position' in obs:
            self.positions.append(obs["position"])

        should_record = self.should_record_last and self.obs_full(obs)
        self.should_record_last = self.should_record(obs)

        # Process transitions
        if self.state == State.STARTING:
            if self.obs_full(obs):
                self.state = State.LEARNING

        elif self.state == State.LEARNING:
            if not self.obs_full(obs):
                self.state = State.ERRORED
            
            if self.is_stuck(obs):
                self.state = State.RECOVERY
                self.recovery_steer = np.random.uniform(-1, 1)
                self.start_recovery_time = rospy.Time.now()
                self.num_recovery += 1
            
            if self.is_inverted(obs):
                self.state = State.CRASHED

        elif self.state == State.CRASHED:
            self.state = State.RECOVERY
            self.recovery_steer = np.random.uniform(-1, 1)
            self.start_recovery_time = rospy.Time.now()
            self.positions.clear()

        elif self.state == State.ERRORED:
            pass

        elif self.state == State.TELEOP or self.state == State.TELEOP_RECORD:
            if rospy.Time.now() - self.last_teleop_time > rospy.Duration(0.5):
                if self.obs_full(obs):
                    self.positions.clear()
                    self.state = State.LEARNING
                else:
                    self.state = State.STARTING

        elif self.state == State.RECOVERY:
            if rospy.Time.now() - self.start_recovery_time > rospy.Duration(1.0):
                if self.obs_full(obs):
                    self.positions.clear()
                    self.state = State.LEARNING
                else:
                    self.state = State.STARTING

        truncated = should_record and (self.state == State.RECOVERY or self.state == State.ERRORED)
        terminated = should_record and (self.state == State.CRASHED or self.state == State.TELEOP)

        self.mode_publisher.publish(stdm.String(self.state.name))

        return should_record, truncated, terminated