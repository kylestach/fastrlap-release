cmake_minimum_required(VERSION 3.0.2)
project(offroad_sensor_fusion)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)
catkin_package(CATKIN_DEPENDS rospy std_msgs nav_msgs geometry_msgs)
catkin_python_setup()

catkin_install_python(PROGRAMS bin/state_estimator_node bin/gps_state_estimator_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    FILES_MATCHING PATTERN "*.launch"
)
