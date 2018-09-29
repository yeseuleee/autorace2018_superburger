cmake_minimum_required(VERSION 2.8.3)
project(signal_lamp_detection)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
   message_generation
  roscpp
  sensor_msgs
  std_msgs
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
 )

###################################
## catkin specific configuration ##
###################################
set(CMAKE_CXX_FLAGS "-fPIE -fPIC -std=c++11 -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -Wformat -Wformat-security -Wall ${CMAKE_CXX_FLAGS}")
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES signal_lamp_detection
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
  DEPENDS system_lib
)
find_package(OpenCV)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library


add_executable(signal_lamp_detection 
src/signal_lamp_detection.cpp)


## Specify libraries to link a library or executable target against
 target_link_libraries(signal_lamp_detection
   ${catkin_LIBRARIES}
   ${OpenCV_INCLUDE_DIRS}
 )

