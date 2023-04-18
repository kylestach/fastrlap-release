cmake_minimum_required(VERSION 3.0.2)
project(tensor_dict_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Tensor.msg
  TensorDict.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
)

catkin_package(CATKIN_DEPENDS message_runtime)