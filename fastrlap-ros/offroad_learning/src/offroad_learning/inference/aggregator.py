import collections

import numpy as np
import jax.numpy as jnp
from PIL import Image

import rospy

import jax
import tf
import tf2_ros
import ros_numpy
import tf2_geometry_msgs as tf2gm
import PyKDL as kdl
import tf.transformations as tft
import time

import sensor_msgs.msg as sm
import geometry_msgs.msg as gm
import nav_msgs.msg as nm

def _pose_to_kdl_transform(pose: gm.Pose):
    return kdl.Frame(
        kdl.Rotation.Quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
        kdl.Vector(pose.position.x, pose.position.y, pose.position.z),
    )

@jax.jit
def _run_encoder(encoder, pixels_stacked):
    return encoder.apply_fn({'params': encoder.params}, pixels_stacked)

class RosDataAggregator:
    def __init__(self, num_stack, image_callback, state_keys, fixed_frame_id, odom_frame_id='odom', encoder=None):
        self.state_keys = state_keys
        self.fixed_frame_id = fixed_frame_id
        self.odom_frame_id = odom_frame_id
        self.encoder = encoder

        self.latest_states = {}
        self.latest_pixels = collections.deque(maxlen=num_stack)
        for _ in range(num_stack):
            self.latest_pixels.append(np.zeros((128, 128, 3)))

        self.image_callback = image_callback
        self.prev_action = None

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.img_subscription = rospy.Subscriber(
            rospy.get_param("~image_topic", "/camera/image_raw"),
            sm.Image,
            self.receive_image,
        )

        self.accel_hist = collections.deque(maxlen=50)

        imu_topic = rospy.get_param("~imu_topic", "")
        accel_topic = rospy.get_param("~accel_topic", "")
        gyro_topic = rospy.get_param("~gyro_topic", "")

        assert (imu_topic != "" and accel_topic == "" and gyro_topic == "") or (imu_topic == "" and accel_topic != "" and gyro_topic != ""), "Must specify either imu_topic or accel_topic and gyro_topic"
        if imu_topic:
            self.imu_subscription = rospy.Subscriber(
                imu_topic, sm.Imu, self.receive_inertial
            )
        else:
            self.accel_subscription = rospy.Subscriber(
                accel_topic, sm.Imu, self.receive_accel
            )
            self.gyro_subscription = rospy.Subscriber(
                gyro_topic, sm.Imu, self.receive_gyro
            )

        self.odom_subscription = rospy.Subscriber(
            rospy.get_param("~odom_topic", "/odom/sample"),
            nm.Odometry,
            self.receive_odometry,
        )

    def observation(self):
        return {
            **self.latest_states,
            "states": np.concatenate(
                [self.latest_states[k] for k in self.state_keys]
            ) if all(k in self.latest_states for k in self.state_keys) else None,
        }

    def receive_image(self, msg: sm.Image):
        # Control rate is 10Hz
        image = ros_numpy.numpify(msg)
        self.latest_pixels.append(np.asarray(Image.fromarray(image).resize((128, 128)).transpose(Image.ROTATE_180)))

        pixels_stacked = np.stack(self.latest_pixels, axis=-1)
        if self.encoder is None:
            self.latest_states["pixels"] = pixels_stacked
        else:
            pixels_stacked = pixels_stacked.reshape((1, *pixels_stacked.shape[:-2], -1)) / 255.0
            self.latest_states["image_embeddings"] = np.array(_run_encoder(self.encoder, pixels_stacked)[0])

        if self.image_callback is not None:
            observation = self.observation()
            self.prev_action = self.image_callback(observation)
            self.prev_observation = observation
            self.latest_states["action"] = self.prev_action 

    def receive_gyro(self, msg: gm.Vector3):
        self.latest_states["gyro"] = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def receive_accel(self, msg: gm.Vector3):
        self.latest_states["accel"] = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        self.accel_hist.append(np.linalg.norm(self.latest_states["accel"][:2]))
        self.latest_states["max_accel_hist"] = np.array([max(self.accel_hist) if len(self.accel_hist) > 0 else 0.0])

    def receive_inertial(self, msg: sm.Imu):
        self.latest_states["gyro"] = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]
        )
        self.latest_states["accel"] = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )
        self.accel_hist.append(np.linalg.norm(self.latest_states["accel"][:2]))
        self.latest_states["max_accel_hist"] = np.array([max(self.accel_hist)])

    def receive_odometry(self, msg: nm.Odometry):
        world_to_msg = _pose_to_kdl_transform(msg.pose.pose).Inverse()
        linear_relative = kdl.Vector(
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z
        )
        angular_relative = kdl.Vector(
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        )
        self.latest_states["relative_linear_velocity"] = np.array(
            [linear_relative.x(), linear_relative.y(), linear_relative.z()]
        )
        self.latest_states["relative_angular_velocity"] = np.array(
            [angular_relative.x(), angular_relative.y(), angular_relative.z()]
        )
        self.latest_states["relative_angular_velocity_z"] = np.array(
            [angular_relative.z()]
        )

        try:
            transform_map_to_base: gm.TransformStamped = self.tf_buffer.lookup_transform(
                self.fixed_frame_id, "base_link", rospy.Time(0), rospy.Duration(0.01)
            )
        except tf.LookupException:
            rospy.logwarn(f"Could not lookup transform from {self.fixed_frame_id} to base_link")
            return

        # try:
        #     transform_map_to_odom: gm.TransformStamped = self.tf_buffer.lookup_transform(
        #         self.odom_frame_id, self.fixed_frame_id, rospy.Time(0), rospy.Duration(0.01)
        #     )
        # except tf.LookupException:
        #     rospy.logwarn(f"Could not lookup transform from {self.fixed_frame_id} to {self.odom_frame_id}")
        #     return

        # try:
        #     transform_base_to_odom: gm.TransformStamped = self.tf_buffer.lookup_transform(
        #         self.odom_frame_id, "base_link", rospy.Time(0), rospy.Duration(0.01)
        #     )
        # except tf.LookupException:
        #     rospy.logwarn(f"Could not lookup transform from {self.odom_frame_id} to base_link")
        #     return

        # Get world-relative position
        self.latest_states["position"] = np.array(
            [
                transform_map_to_base.transform.translation.x,
                transform_map_to_base.transform.translation.y,
                transform_map_to_base.transform.translation.z,
            ]
        )
        self.latest_states["orientation"] = np.array(
            [
                transform_map_to_base.transform.rotation.x,
                transform_map_to_base.transform.rotation.y,
                transform_map_to_base.transform.rotation.z,
                transform_map_to_base.transform.rotation.w,
            ]
        )
        self.latest_states["pose_2d"] = np.array([
            transform_map_to_base.transform.translation.x,
            transform_map_to_base.transform.translation.y,
            tft.euler_from_quaternion([
                transform_map_to_base.transform.rotation.x,
                transform_map_to_base.transform.rotation.y,
                transform_map_to_base.transform.rotation.z,
                transform_map_to_base.transform.rotation.w,
            ])[-1]
        ])

        # Copies in the odom frame
        # self.latest_states["pose_2d_odom"] = np.array([
        #     transform_base_to_odom.transform.translation.x,
        #     transform_base_to_odom.transform.translation.y,
        #     tft.euler_from_quaternion([
        #         transform_base_to_odom.transform.rotation.x,
        #         transform_base_to_odom.transform.rotation.y,
        #         transform_base_to_odom.transform.rotation.z,
        #         transform_base_to_odom.transform.rotation.w,
        #     ])[-1]
        # ])

        # goals_odom = []
        # if "goal_absolute" in self.latest_states:
        #     goal = self.latest_states["goal_absolute"]
        #     goal_odom = tf2gm.do_transform_point(gm.PointStamped(msg.header, gm.Point(goal[0], goal[1], goal[2])), transform_map_to_odom).point
        #     goals_odom.append(np.array([goal_odom.x, goal_odom.y, goal_odom.z]))
        #     self.latest_states["goal_odom"] = np.concatenate(goals_odom, axis=0)

    def receive_goal(self, msg: gm.PoseArray):
        transform = self.tf_buffer.lookup_transform(
            "base_link", msg.header.frame_id, rospy.Time(0), rospy.Duration(0.01)
        )
        goals = []
        for pose in msg.poses:
            self.latest_states["goal_absolute"] = np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ]
            )
            goal_relative = tf2gm.do_transform_pose(
                gm.PoseStamped(msg.header, pose), transform
            ).pose.position
            distance_to_goal = np.sqrt(goal_relative.x ** 2 + goal_relative.y ** 2)
            goals.append(np.array([goal_relative.x / distance_to_goal, goal_relative.y / distance_to_goal, distance_to_goal]))
        self.latest_states["goal_relative"] = np.concatenate(goals)
