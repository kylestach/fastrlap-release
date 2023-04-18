import rospy
import tf2_ros
import numpy as np

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_srvs.srv import Empty, EmptyResponse
import sensor_msgs.msg as sm

class RosGoalGraph:
    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.fixed_frame_id = rospy.get_param("~fixed_frame")

        self.goal_loop = []
        self.capture_goal = rospy.Service("/offroad_learning/capture_goal", Empty, self.capture_goal_callback)
        self.goal_pub = rospy.Publisher("/offroad_learning/recorded_goal", PoseArray, queue_size=1)
        self.joy_sub = rospy.Subscriber("/vesc/joy", sm.Joy, self.joy_callback)
        self.button_was_pressed = False

    def joy_callback(self, joy):
        if joy.buttons[1] == 1 and not self.button_was_pressed:
            self.capture_goal_callback(None)
            rospy.logwarn(f"Goals: {self.goal_loop}")
        self.button_was_pressed = (joy.buttons[1] == 1)

    def capture_goal_callback(self, _):
        # Wait for the map to become ready
        try:
            tx = self.buffer.lookup_transform(self.fixed_frame_id, "base_link", rospy.Time.now(), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Cannot capture goal; transform failed due to {e}")
            return

        position = np.array([tx.transform.translation.x, tx.transform.translation.y])
        self.goal_loop.append(position)

        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = rospy.Time.now()
        for goal in self.goal_loop:
            pose = Pose()
            pose.position.x = goal[0]
            pose.position.y = goal[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.goal_pub.publish(pose_array)

        return EmptyResponse


def main():
    rospy.init_node("goal_graph_recorder")
    goal_graph = RosGoalGraph()
    rospy.spin()

if __name__ == "__main__":
    main()
