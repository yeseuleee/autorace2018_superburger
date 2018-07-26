cmake_minimum_required(VERSION 2.8.3)
project(control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  laser_geometry
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  message_generation
)


################################################
## Declare ROS messages, services and actions ##
################################################



## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )


## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES std_msgs)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES control
  CATKIN_DEPENDS laser_geometry pcl_conversions pcl_ros roscpp sensor_msgs std_msgs tf message_generation
  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#add_dependencies :  빌드하려는 패키지가 메시지나 서비스, dynamic_reconfigure 파라미터를 생성하거나 이를 이용하는 실행 파일을 만드는 경우, add_dependencies()를 이용하여 자동 생성되는 메시지, 서비스, dynamic_reconfigure target에 대한 패키지의 의존성을 추가
#add_dependencies(
#  motor_control_node 
#  control_generate_messages_cpp
#)


###################################
## main control node ##
###################################
#add_executable : 빌드 후 생성할 실행 파일에 대한 옵션 지정(cpp는 여러개 이어서 추가가능, 실행파일은 1개씩만)
add_executable(
  motor_control_node 
  src/motor_control.cpp
  )
#target_link_libraries : 지정 실행 파일을 생성하기에 앞서 링크해야하는 라이브러리나 실행파일 지정
target_link_libraries(
  motor_control_node
   ${catkin_LIBRARIES}
  )
  


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARIY_DIRS})
add_definitions(${PCL_DEFINITIONS})
