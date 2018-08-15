# autorace2018_superburger

## wifi password

5g wifi passwords: ssu851213

## turtle bot id & passwd

id: seuleee

password: 2015031322

## 세팅 파일 (bashrc)

각자 ip주소 알아내고

turtle bot 의 ~/.bashrc 에서

ROS_MASTER_URI 가 노트북 ip

ROS_HOSTNAME 가 turtle bot ip

master pc 의 ~/.bashrc 에는

ROS_MASTER_URI, ROS_HOSTNAME 둘다 자신의 ip를 담는다.

## roslaunch (master pc)

1번 roscore 실행

$ roscore

4번 받아온 영상을 토대로 모터제어를 결정함

$ roslaunch vision_launch vision_launch.launch

5번 모터 컨트롤

$ roslaunch control control.launch

## roslaunch (turtle bot)

2번. turtle bot의 카메라에서 받아온 영상을 master pc로 스트리밍.

$ roslaunch camera_image camera_image.launch

3번. 모터제어에 관한 토픽정보를 제공

$ roslaunch turtlebot3_bringup turtlebot3_robot.launch
