cmake_minimum_required(VERSION 3.0.2)
project(tensor_dict_convert)

find_package(catkin REQUIRED COMPONENTS rospy tensor_dict_msgs)

catkin_python_setup()

catkin_package(
 CATKIN_DEPENDS rospy
)