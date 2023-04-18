import collections
from itertools import cycle
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import json

from geometry_msgs.msg import Pose, PoseArray, PoseStamped


NUM_GOALS = 1
GOAL_THRESHOLD = 3.0


class GoalGraph:
    def __init__(self, fixed_frame_id):
        self.fixed_frame_id = fixed_frame_id

        # self.goal_iter = cycle([
        #     (492818.58292537474, 5527490.252053209),
        #     (492787.05475763197, 5527415.802657225),
        #     (492808.597964718, 5527480.553456435),
        #     (492774.2589048771, 5527541.41242493),
        #     (492855.9575058973, 5527506.0054903515),
        # ])
        # self.goal_iter = cycle([
        #     (492818.47709581, 5527516.79028953),
        #     (492824.00192887, 5527517.86657135),
        #     (492824.43477213, 5527524.50892515),
        #     (492821.13319653, 5527523.92347255),
        #     (492815.75683166, 5527523.02542683),
        #     (492813.03048466, 5527521.64724148),
        #     (492815.62690636, 5527516.78011265),
        # ])
        # self.goal_iter = cycle([
        #     (8.0, 0.0),
        #     (0.0, 0.0),
        # ])
        # self.goal_iter = cycle([
        #     ( 6.28102807, -0.67434333),
        #     (15.46101333, -0.39252481),
        #     (16.85033942, -1.66686692),
        #     (16.58029443, -8.00996932),
        #     ( 9.17652327, -8.73401455),
        #     (-1.33585273, -8.86451632),
        #     (-4.11966963, -3.54762611),
        #     (-0.54993951, -0.76961602),
        # ])
        # self.goal_iter = cycle(enumerate([
        #     (5.33418263, 0.0911337 ),
        #     (6.22691778, 4.37476528),
        #     (0.38207896, 4.49907576),
        #     (-0.0266515,  0.4914248)
        # ]))
        # self.goal_iter = cycle(enumerate([
        #     (2.10269403, 0.34488638),
        #     (3.12799894, 3.18391499),
        #     (0.27073666, 4.05643371),
        #     (-0.04740736,  1.12595609),
        # ]))
        # self.goal_iter = cycle(enumerate([
        #     (-9.90928459, -9.49233878),
        #     (-0.432289  , -2.63320777),
        #     (-2.21655691, 13.94887409),
        #     ( -8.3367129 , -10.61985777),
        # ]))
        # self.goal_iter = cycle(enumerate([
        #     (1.16780979, -5.40302549),
        #     (9.20317706, -1.36299561),
        #     (5.09396803, 16.71493986),
        # ]))
        self.goal_iter = cycle(enumerate([
            (-6.05664119, -10.20149303),
            (-33.02459595,  -8.89588128),
        ]))

        # Cory 1
        self.goal_iter = cycle(enumerate([
            (7.48558315, 0.79947734),
            ( 9.13847626, -28.44004921),
            (-21.7779718 , -29.41927294),
            (-21.88073551,  -0.07610475)
        ]))

        # Soda 5
        self.goal_iter = cycle(enumerate([
            (-11.16263003,  -0.82212528),
            (11.93014813, -0.27265652),
            ( 10.53240489, -16.27013002),
            (-10.40514569, -14.67150921),
        ]))

        # BWW Commons
        self.goal_iter = cycle(enumerate([
            (-0.02070998,  0.0428383),
            ( 1.55971676, -4.04448367),
            (-1.08771442, -3.50781278),
        ]))

        # RFS back and forth
        self.goal_iter = cycle(enumerate([
            ( 2.21087147, -7.13730187),
            (-26.47729994,   1.20538399),
        ]))

        # BWW outside back and forth
        self.goal_iter = cycle(enumerate([
            (0.25281849, 2.04047412),
            ( 7.38172911, -9.62627015)
        ]))

        # BWW8 Conference Room
        self.goal_iter = cycle(enumerate([
            [4.85715404, 2.01852784],
            [1.40373069, 4.36832277],
            [0.23860978, 0.64367575],
        ]))

        # Red statue thing
        self.goal_iter = cycle(enumerate([
            (221.34486764, -73.70071795),
            (228.20400522, -62.36224633),
        ]))

        # Barker - UTM zero is (564649.74, 4191966.66)
        self.goal_iter = cycle(enumerate([
            (-47.68940611,  90.17617582),
            (-62.37943966,  86.53864195),
            (-47.36252993,  89.99285564)
        ]))

        # RFS Bathroom Loop
        self.goal_iter = cycle(enumerate([
            (-11.59092659,   3.64889312),
            (-0.06337945, 16.75068065),
            ( 3.31705014, -1.34583597),
        ]))

        # RFS Woodchipper
        self.goal_iter = cycle(enumerate([
            (-29.155151684419252, 11.567492874339223),
            (-14.187522422289476, 20.407110924832523),
            (-15.020810227608308, 10.359588176943362),
            (-14.118506628321484, 4.662815337069333),
        ]))

        # 4.0 Hill
        self.goal_iter = cycle(enumerate([
            (-6.964784860960208, -10.672565669752657),
            (-18.67062807001639, -15.675674008671194),
            (-42.80214678810444, 12.443363225553185),
            (-4.372213858528994, -8.69649629574269),
        ]))

        # Hertz Hall
        self.goal_iter = cycle(enumerate([
            (-3.862860521301627, 4.642431508284062),
            (3.9308972825529054, -7.446089910343289),
            (24.92767575115431, -27.544054055586457),
            (36.364588620490395, -26.64689285028726),
        ]))

        self.start = next(self.goal_iter)[1]
        self.goal_queue = collections.deque(maxlen=NUM_GOALS)

        while len(self.goal_queue) < NUM_GOALS:
            self.goal_queue.append(next(self.goal_iter))
        
        self.goal_graph_pub = rospy.Publisher(
            rospy.get_param("~goal_topic", "/offroad_learning/goal"), PoseArray, queue_size=1
        )
        self.next_goal_pub = rospy.Publisher(
            "/offroad_learning/next_goal", PoseStamped, queue_size=1
        )

        self.lap_times = []
        self.last_lap_start = rospy.Time.now()


    def goal_poses(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.fixed_frame_id
        pose_array.header.stamp = rospy.Time.now()

        for _, goal in self.goal_queue:
            pose = Pose()
            pose.position.x = goal[0]
            pose.position.y = goal[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        return pose_array
    
    def publish(self):
        poses = self.goal_poses()
        self.goal_graph_pub.publish(poses)
        self.next_goal_pub.publish(PoseStamped(poses.header, poses.poses[0]))
    
    def tick(self, robot_position):
        finished_lap = False
        if np.linalg.norm(robot_position[:2] - self.goal_queue[0][1]) < GOAL_THRESHOLD:
            lap_time = (rospy.Time.now() - self.last_lap_start).to_sec()
            if self.goal_queue[0][0] == 1:
                self.lap_times.append(lap_time)
                self.last_lap_start = rospy.Time.now()
                print(f"Lap times so far: {self.lap_times}")
                finished_lap = True

            self.start = self.goal_queue[0]
            self.goal_queue.popleft()

            self.goal_queue.append(next(self.goal_iter))

            self.publish()

            return True, finished_lap, lap_time

        self.publish()
        return False, finished_lap, None


class RosGoalGraph:
    def __init__(self, fixed_frame_id):
        self.fixed_frame_id = fixed_frame_id

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.goal_queue = []
        self.goal_pub = rospy.Publisher(
            rospy.get_param("~goal_topic", "/offroad_learning/goal"), PoseArray, queue_size=1
        )
        self.next_goal_pub = rospy.Publisher(
            "/offroad_learning/next_goal", PoseStamped, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(0.05), self.goal_timer_callback)
        self.goal_iter = self.next_goal()

    def next_goal(self):
        return cycle([
            (492818.58292537474, 5527490.252053209),
            (492787.05475763197, 5527415.802657225),
            (492808.597964718, 5527480.553456435),
            (492774.2589048771, 5527541.41242493),
            (492855.9575058973, 5527506.0054903515),
        ])

    def goal_timer_callback(self, event):
        # Wait for the map to become ready
        try:
            tx = self.buffer.lookup_transform(self.fixed_frame_id, "base_link", rospy.Time.now(), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return

        position = np.array([tx.transform.translation.x, tx.transform.translation.y])

        if len(self.goal_queue) == 0:
            self.goal_queue.append(next(self.goal_iter))

        while len(self.goal_queue) < NUM_GOALS:
            self.goal_queue.append(next(self.goal_iter))

        if np.linalg.norm(position - self.goal_queue[0]) < 1.0:
            self.goal_queue.pop(0)
            self.goal_queue.append(next(self.goal_iter))

        pose_array = PoseArray()
        pose_array.header.frame_id = self.fixed_frame_id
        pose_array.header.stamp = rospy.Time.now()
        for goal in self.goal_queue:
            pose = Pose()
            pose.position.x = goal[0]
            pose.position.y = goal[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.goal_pub.publish(pose_array)
        self.next_goal_pub.publish(PoseStamped(pose_array.header, pose_array.poses[0]))


def main():
    rospy.init_node("goal_graph")
    goal_graph = RosGoalGraph()
    rospy.spin()

if __name__ == "__main__":
    main()
