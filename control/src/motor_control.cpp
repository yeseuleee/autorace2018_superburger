#include "ros/ros.h"
#include <iostream>
#include "sensor_msgs/JointState.h"
#include "sensor_msgs/PointCloud2.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Float64.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float32MultiArray.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

sensor_msgs::JointState js;
pcl::PointCloud<pcl::PointXYZI> pc;//i is intensity by lidar
bool right = true;

bool js_init_checked = false;
std::vector<float> current_js;
std::vector<float> prev_js;

#define WHEEL_RADIUS 0.033
#define WHEEL_BASE_LENGTH 0.160 //burger( waffle : 0.287)
float odm_theta = 0;
float odm_x = 0;
float odm_y = 0; 
void update_odometry(){

    float delta_right_rad, delta_left_rad, m_per_rad;
	float dr, dl, dc;
	float x_new, y_new, theta_new;
	float x, y, theta;
	float rad;

	rad = 180/M_PI;
	if(!current_js.empty()){
        x = odm_x;
        y = odm_y;
        theta = odm_theta;

        //compute odometry here
        delta_left_rad = current_js[0] - prev_js[0];
        delta_right_rad = current_js[1] - prev_js[1];
        
        m_per_rad = (2*M_PI*WHEEL_RADIUS)*(rad/360);

        dr = m_per_rad * delta_right_rad;
        dl = m_per_rad * delta_left_rad;
        dc = (dr+dl)/2;

        x_new = x + dc*cos(theta);
        y_new = y + dc*sin(theta);
        theta_new = theta + (dr - dl)/WHEEL_BASE_LENGTH;

        //new 2017-10-25
        theta_new = atan2(sin(theta_new),cos(theta_new)); //shoud be -pi ~ pi

        //save the wheel encoder rad for the next estimate
        prev_js[0] = current_js[0];
        prev_js[1] = current_js[1];
        

        //update your estimate of (x,y,theta)
        odm_x = x_new;
        odm_y = y_new;
        odm_theta = theta_new;   
    }
    
}
//pcl::PointCloud<pcl::PointXYZ> pc;
void jointCallback(const sensor_msgs::JointState& msg) { 
    js.position = msg.position;//1행 2열의 벡터(letf,right)
    std::cout<<"js.position : "<<js.position[0] <<"," <<js.position[1]<<std::endl;
    current_js.clear();
    current_js.push_back(js.position[0]);
    current_js.push_back(js.position[1]);
    if(!js_init_checked){
        prev_js.clear();
        prev_js.push_back(js.position[0]);
        prev_js.push_back(js.position[1]);
        js_init_checked = true;
    }
    update_odometry();
    std::cout<<"odm_x : "<<odm_x<<std::endl;
    std::cout<<"odm_y : "<<odm_y<<std::endl;
    std::cout<<"odm_theta : "<<odm_theta<<std::endl;
}
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) 
{
	//if(msg->header.frame_id != "base_scan") ROS_ERROR("HLDS DRIVER SHOULD WORK!!");//scan데이터가 발행되는 frame_id
	//if(msg->header.frame_id != "base_link") ROS_ERROR("HLDS DRIVER SHOULD WORK!!");
    //else pcl::fromROSMsg(*msg,pc);
    //pcl::fromROSMsg(*msg,pc);
}
void pidCallback(const std_msgs::Float32MultiArray::ConstPtr& goalData){
    // float u_x, u_y; //reference x , y
	// float theta_g; //theta goal (theta between goal and ev3)
	// float e_k,e_P,e_I,e_D; //PID errors
	// float w; //omega

	// u_x = in.x_g - odm_x;
	// u_y = in.y_g - odm_y;
	// theta_g = atan2(u_y,u_x);
	// e_k = theta_g - odm_theta;
	// e_k = atan2(sin(e_k),cos(e_k));
	// //printf("e_k = %lf\n",e_k);
	// e_P = e_k;
	// e_I = E_k + e_k*0.1; //dt
	// e_D = (e_k - e_k_1)/0.1; //dt

	// w = kp*e_P + ki*e_I + kd*e_D;
	// E_k = e_I;
	// e_k_1 = e_k;

	// out.o_v = in.v_g;
	// out.o_w = w;
	// //printf("u_y , u_x , theta_g = %lf, %lf, %lf\n",u_y,u_x,theta_g);
	// printf("out.o_v = %lf, out.o_w = %lf\n",out.o_v,out.o_w);
	// if(check_event(odm)) return 1;
	// return 0;
}

int main(int argc, char **argv){
    ros::init(argc, argv, "motor_control");
    ros::NodeHandle nh;
    ros::Rate loop_rate(30);

    ros::Publisher twist_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel",1000); 
	ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_out",1000);
	ros::Subscriber state_reader = nh.subscribe("/joint_states",1000,jointCallback); 
	ros::Subscriber point_reader = nh.subscribe("/cloud_points",1000,cloudCallback);
    ros::Subscriber ang_vel_reader = nh.subscribe("/main/angular_vel",1000,pidCallback);
	//ros::ServiceServer control_srv = nh.advertiseService("/msrv",exec_command);
	
    geometry_msgs::Twist twist_cmd;
	twist_cmd.linear.x = twist_cmd.linear.y = twist_cmd.linear.z = 0; //twist initialization
	twist_cmd.angular.x = twist_cmd.angular.y = twist_cmd.angular.z = 0;
    js.position.push_back(0.0);
    //std::cout <<"hu"<<std::endl;
    
    while(ros::ok()){  
        ros::spinOnce(); //call all callback functions
        
        //if(right){

        //twist_cmd.linear.y = 0.1;
        //twist_cmd.angular.z = -0.1;
        //}
        twist_pub.publish(twist_cmd);
        loop_rate.sleep();
    }
    return 0;
}