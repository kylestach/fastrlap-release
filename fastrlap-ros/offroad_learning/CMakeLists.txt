cmake_minimum_required(VERSION 3.0.2)
project(offroad_learning)
find_package(catkin REQUIRED rospy tensor_dict_msgs)
catkin_package(CATKIN_DEPENDS rospy tensor_dict_msgs)
catkin_python_setup()

catkin_install_python(PROGRAMS bin/training_node bin/inference_node bin/goal_graph_recorder_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)