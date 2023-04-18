import rospy
import sensor_msgs.msg as sm
import geometry_msgs.msg as gm
import nav_msgs.msg as nm
import std_msgs.msg as stdm
import tf2_ros
import tf2_geometry_msgs as tf2gm

import numpy as np

import pygeodesy as geodesy


class GpsStateEstimationNode:
    def __init__(self):
        self.utm_datum = None

        self.last_gps_time = None
        self.last_gps_vel_time = None
        self.last_update_time = None
        self.last_utm = None
        self.last_vel = None
        self.last_input = np.zeros((2,))

        self.dt = rospy.get_param("~dt", 0.05)
        self.x = np.ones((1,))
        self.P = np.identity(1) * 1e3

        self.gps_subscription = rospy.Subscriber(
            "gps/fix", sm.NavSatFix, self.gps_callback)
        self.gps_vel_subscription = rospy.Subscriber(
            "gps/vel", gm.TwistWithCovarianceStamped, self.gps_vel_callback)
        self.wheel_odom_subscription = rospy.Subscriber(
            "wheel_odom", nm.Odometry, self.wheel_odom_callback)
        self.gyro_subscription = rospy.Subscriber(
            "imu/data", sm.Imu, self.gyro_callback)

        self.odom_publisher = rospy.Publisher(
            "odom", nm.Odometry, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def observe(self, x: np.ndarray, input: np.ndarray):
        # Observation model is on world-space velocity
        heading = x[0]
        input_speed = input[0]
        world_vel = np.array([
            np.cos(heading),
            np.sin(heading),
        ]) * input_speed

        return world_vel

    def observe_jacobian(self, x: np.ndarray, input: np.ndarray):
        # Observation model is on world-space velocity
        heading = x[0]
        input_speed = input[0]

        H_x = np.array([
            [-np.sin(heading)],
            [np.cos(heading)],
        ]) * input_speed
        H_u = np.array([
            [np.cos(heading), 0.0],
            [np.sin(heading), 0.0],
        ])
        return H_x, H_u

    def predict(self, x: np.ndarray, input: np.ndarray, dt):
        input_angular_speed = input[1]
        return x + input_angular_speed * dt

    def predict_jacobian(self, x: np.ndarray, input: np.ndarray, dt):
        return np.ones((1, 1)), np.array([0, dt])

    def do_filter(self, x: np.ndarray, u: np.ndarray, obs: np.ndarray, P: np.ndarray, dt: float):
        # Apply the Kalman filter
        input_speed = u[0]
        Q = np.diag([0.01])
        R = np.diag([1.0, 1.0])

        # Predict
        x_pred = self.predict(x, u, dt)
        F_x, F_u = self.predict_jacobian(x, u, dt)
        P_pred = F_x @ P @ F_x.T + Q

        # Update
        z_pred = self.observe(x_pred, u)
        H_x, H_u = self.observe_jacobian(x_pred, u)
        S = H_x @ P_pred @ H_x.T + R
        K = P_pred @ H_x.T @ np.linalg.inv(S)
        x = x_pred + K @ (obs - z_pred)
        P = (np.identity(1) - K @ H_x) @ P_pred
        P = (P + P.T) / 2

        return x, P

    def gps_callback(self, msg: sm.NavSatFix):
        # Convert to UTM
        if self.utm_datum is None:
            utm: geodesy.Utm = geodesy.utm.toUtm8(msg.latitude, msg.longitude)
            self.utm_datum = utm.datum
        else:
            utm = geodesy.utm.toUtm8(
                msg.latitude, msg.longitude, datum=self.utm_datum)

        self.position_utm = np.array([utm.easting, utm.northing])

    def gps_vel_callback(self, msg: gm.TwistWithCovarianceStamped):
        self.last_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

        if self.last_gps_vel_time is None:
            dt = 0.0
        else:
            dt = (msg.header.stamp - self.last_gps_vel_time).to_sec()
        self.last_gps_vel_time = msg.header.stamp

        self.x, self.P = self.do_filter(
            self.x, self.last_input, self.last_vel, self.P, dt)

        # Publish an Odometry message
        forward_velocity_from_gps = np.cos(self.x[0]) * msg.twist.twist.linear.x + np.sin(self.x[0]) * msg.twist.twist.linear.y
        self.odom_publisher.publish(nm.Odometry(
            header=stdm.Header(
                stamp=rospy.Time.now(),
                frame_id="utm",
            ),
            child_frame_id="base_link",
            pose=gm.PoseWithCovariance(
                pose=gm.Pose(
                    position=gm.Point(
                        x=self.position_utm[0],
                        y=self.position_utm[1],
                        z=0.0,
                    ),
                    orientation=gm.Quaternion(
                        x=0.0,
                        y=0.0,
                        z=np.sin(self.x[0] / 2),
                        w=np.cos(self.x[0] / 2),
                    ),
                ),
                covariance=np.diag([1.0, 1.0, 1.0, 0.1, 0.1, self.P[0, 0]]).flatten().tolist(),
            ),
            twist=gm.TwistWithCovariance(
                twist=gm.Twist(
                    linear=gm.Vector3(
                        x=forward_velocity_from_gps,
                        y=0.0,
                        z=0.0,
                    ),
                    angular=gm.Vector3(
                        x=0.0,
                        y=0.0,
                        z=self.last_input[1],
                    ),
                ),
                covariance=(1e-2 * np.eye(6)).flatten().tolist(),
            ),
        ))

        # Publish a TF message
        self.tf_broadcaster.sendTransform(
            gm.TransformStamped(
                header=stdm.Header(
                    stamp=rospy.Time.now(),
                    frame_id="utm",
                ),
                child_frame_id="base_link",
                transform=gm.Transform(
                    translation=gm.Vector3(
                        x=self.position_utm[0],
                        y=self.position_utm[1],
                        z=0.0,
                    ),
                    rotation=gm.Quaternion(
                        x=0.0,
                        y=0.0,
                        z=np.sin(self.x[0] / 2),
                        w=np.cos(self.x[0] / 2),
                    ),
                ),
            ),
        )

    def wheel_odom_callback(self, msg: nm.Odometry):
        self.last_input[0] = msg.twist.twist.linear.x

    def gyro_callback(self, msg: sm.Imu):
        # Convert to base_link frame
        # try:
        #     imu_to_base = self.tf_buffer.lookup_transform(
        #         "base_link", msg.header.frame_id, rospy.Time(0))
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     rospy.logwarn(f"Failed to lookup transform from base_link to {msg.header.frame_id}")
        #     return

        # angular_velocity = tf2gm.do_transform_vector3(
            # msg.angular_velocity, imu_to_base)
        self.last_input[1] = msg.angular_velocity.z

def main():
    rospy.init_node("gps_ekf")
    ekf = GpsStateEstimationNode()
    rospy.spin()

if __name__ == "__main__":
    main()