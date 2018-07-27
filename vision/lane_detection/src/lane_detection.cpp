#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <lane_detection_func/lane_detection_func.hpp>
#include "sensor_msgs/PointCloud2.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_search.h>
#include <pcl/ModelCoefficients.h>
#include <pcl_ros/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl_ros/segmentation/sac_segmentation.h>
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float32MultiArray.h"
#include <time.h>
#include <string>
#include <math.h>

#define COORDI_COUNT 4000
#define CLOCK_PER_SEC 1000
static const std::string OPENCV_WINDOW_VF = "Image by videofile";
static const std::string OPENCV_WINDOW_WC = "Image by webcam";
static int debug;
// 기본 영상, 디버깅 메세지 출력
static int web_cam;
// true -> 웹캠영상 / false -> 비디오파일
static int imshow;
// 이미지 프로세싱 중간 과정 영상 출력
static int track_bar;
// 트랙바 컨트롤
static int time_check;
// ?
static int lable;

static int gazebo;
static int bird_eye_view;
static int auto_shot;
static int auto_record;
static int for_gui;
// 횡단보도 탐지방법 찾기
static const std::string record_name;

static int left_min_interval, left_max_interval;
static int right_min_interval, right_max_interval;
static float right_rotation_y_goal,left_rotation_y_goal;
static float default_x_goal, default_y_goal;

static int y_hmin, y_hmax, y_smin, y_smax, y_vmin, y_vmax;
static int w_hmin, w_hmax, w_smin, w_smax, w_vmin, w_vmax;

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
double fps = 3;
int control_first = 0;
int fourcc = CV_FOURCC('X','V','I','D'); // codec
bool isColor = true;
cv::Point center_pt_t, center_pt_b;

cv::VideoWriter video_left;
cv::VideoWriter video_right;
cv::VideoWriter video_main;

static std::string groupName;

lane_detect_algo::vec_mat_t lane_m_vec;

float left_interval, right_interval;
float left_ang_vel, right_ang_vel;
float x_goal_, y_goal_, prev_y_goal_;
float left_theta = 0;
float right_theta = 0;
double test[10] = {0,0,0.1,0,0,0,0.1,0,0,0};
int test_num = 0;



using namespace lane_detect_algo;
using namespace std;


class InitImgObjectforROS {

public:
        ros::NodeHandle nh;
        image_transport::ImageTransport it;
        image_transport::Subscriber sub_img;
        ros::Subscriber depth_sub;
        std_msgs::Int32MultiArray coordi_array;
        std_msgs::Float32MultiArray goal_array;
        std::vector<int> lane_width_array;
        //cv::Mat pub_img;
        ros::Publisher pub = nh.advertise<std_msgs::Int32MultiArray>("/"+groupName+"/lane",100);//Topic publishing at each camera
        ros::Publisher goal_pub = nh.advertise<std_msgs::Float32MultiArray>("/"+groupName+"/pixel_goal",100);
        ros::Publisher ang_vel_pub = nh.advertise<std_msgs::Float32MultiArray>("/"+groupName+"/angular_vel",100);
        int imgNum = 0;//for saving img
        cv::Mat output_origin_for_copy;//for saving img
        InitImgObjectforROS();
        ~InitImgObjectforROS();
        void depthMessageCallback(const sensor_msgs::PointCloud2::ConstPtr& input);
        void imgCb(const sensor_msgs::ImageConstPtr& img_msg);
        void initParam();
        void initMyHSVTrackbar(const string &trackbar_name);
        void setMyHSVTrackbarValue(const string &trackbar_name);
        void setColorPreocessing(lane_detect_algo::CalLane callane, cv::Mat src, cv::Mat& dst_y, cv::Mat& dst_w);
        void setProjection(lane_detect_algo::CalLane callane, cv::Mat src, unsigned int* H_aix_Result_color);
        void restoreImgWithLangeMerge(lane_detect_algo::CalLane callane, cv::Mat origin_size_img, cv::Mat src_y, cv::Mat src_w, cv::Mat& dst);
        void extractLanePoint(cv::Mat origin_src, cv::Mat lane_src);
        void initMyHSVTrackbar_old(const string &trackbar_name, int *hmin, int *hmax, int *smin, int *smax, int *vmin, int *vmax);
        void setMyHSVTrackbarValue_old(const string &trackbar_name,int *hmin, int *hmax, int *smin, int *smax, int *vmin, int *vmax);
        void setPixelGoal(double* goal, int num);
};



InitImgObjectforROS::InitImgObjectforROS() : it(nh){
        initParam();
        if(!web_cam) {//'DEBUG_SW == TURE' means subscribing videofile image
                sub_img = it.subscribe("/"+groupName+"/videofile/image_raw",1,&InitImgObjectforROS::imgCb,this);
        }
        else{         //'DEBUG_SW == FALSE' means subscribing webcam image
                if(!gazebo){//use webcam topic
                        sub_img = it.subscribe("/"+groupName+"/raw_image",1,&InitImgObjectforROS::imgCb,this);
                        //sub_img = it.subscribe("/camera/depth/points",1,&InitImgObjectforROS::imgCb,this);
                        //depth_sub = nh.subscribe("/camera/depth_registered/points", 1, &InitImgObjectforROS::depthMessageCallback, this);
                }
                else{//use gazebo topic
                        sub_img = it.subscribe("/camera/image",1,&InitImgObjectforROS::imgCb,this);
                }
                
                
               
        }

        if(track_bar) {
                initMyHSVTrackbar(groupName+"_YELLOW_TRACKBAR");
                initMyHSVTrackbar(groupName+"_WHITE_TRACKBAR");
        }
}


InitImgObjectforROS::~InitImgObjectforROS(){
        if(debug) {//'DEBUG_SW == TURE' means subscribing videofile image
                cv::destroyWindow(OPENCV_WINDOW_VF);
        }
        else{     //'DEBUG_SW == FALE' means subscribing webcam image
                cv::destroyWindow(OPENCV_WINDOW_WC);
        }
}
void InitImgObjectforROS::depthMessageCallback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
      // input.x;
//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//      pcl::fromROSMsg (*input, *cloud);
//      cloud.resize(320,240);
//      cv::Mat imageFrame;
//      if (cloud->isOrganized()) {
//         imageFrame = cv::Mat(cloud->height, cloud->width, CV_8UC3); 

//         for (int h=0; h<imageFrame.rows; h++) {
//             for (int w=0; w<imageFrame.cols; w++) {

//                 pcl::PointXYZRGB point = cloud->at(w, h);

//                 Eigen::Vector3i rgb = point.getRGBVector3i();

//                 imageFrame.at<cv::Vec3b>(h,w)[0] = rgb[2];
//                 imageFrame.at<cv::Vec3b>(h,w)[1] = rgb[1];
//                 imageFrame.at<cv::Vec3b>(h,w)[2] = rgb[0];
                
//                 //int i = centre_x + centre_y*cloud->width;
//                 //depth[call_count] = (float)cloud->points[i].z;

//             }
//          }
//          cv::imshow("aaaaa",imageFrame);
//      }
}


void InitImgObjectforROS::imgCb(const sensor_msgs::ImageConstPtr& img_msg){
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat frame, yellow_hsv, white_hsv, yellow_labeling,white_labeling, laneColor, origin, mergelane, rec_img;
        std::vector<cv::Point> box_pt_y,box_pt_w;
        cv::Point left_slope, right_slope;
        uint frame_height, frame_width;
        bool is_left_box_true = false, is_right_box_true = false;

        //for capture avi
      
        // if(!video->open("/home/seuleee/autorace_video_src/result.avi", fourcc, fps, cv::Size(640/2,480/2), isColor)){
        // delete video;
        // return;
        // }//for capture avi
        
        try{
                cv_ptr = cv_bridge::toCvCopy(img_msg,sensor_msgs::image_encodings::BGR8);
                frame = cv_ptr->image;
                origin = cv_ptr->image;
                rec_img = cv_ptr->image;
                //cv::resize(origin, origin, cv::Size(origin.cols,origin.rows),0,0,CV_INTER_AREA);
                if(!frame.empty()) {
                        if(!gazebo){//if you use gazebo topic than this condition not used.
                                cv::resize(origin,frame,cv::Size(origin.cols/2,origin.rows/2),0,0,CV_INTER_AREA);//img downsizing //320 240
                        }
                        if(auto_record){
                                if(groupName == "left")
                                        video_left << frame;
                                else if(groupName == "right")
                                        video_right << frame;
                                else if(groupName == "main")
                                        video_main << frame;        
                        }                   
                        
                        /*another solution->*/ //cv::resize(frame, frame, cv::Size(), 0.2, 0.2 320 240);
                        frame_height = (uint)frame.rows;
                        frame_width = (uint)frame.cols;
                        

                        lane_detect_algo::CalLane callane;
                        unsigned int* H_yResultYellow = new unsigned int[frame_width];
                        std::memset(H_yResultYellow, 0, sizeof(uint) * frame_width);
                        unsigned int* H_yResultWhite = new unsigned int[frame_width];
                        std::memset(H_yResultWhite, 0, sizeof(uint) * frame_width);
                        unsigned int* H_xResultYellow = new unsigned int[frame_height];
                        std::memset(H_xResultYellow, 0, sizeof(uint) * frame_height);
                        unsigned int* H_xResultWhite = new unsigned int[frame_height];
                        std::memset(H_xResultWhite, 0, sizeof(uint) * frame_height);

                        
                        /////////////////////////

                        ////*Process color detection including trackbar setting*////
                        setColorPreocessing(callane, frame, yellow_hsv, white_hsv);
                        cv::Mat yellow_sobel_v;// = yellow_hsv.clone();
                        cv::Mat yellow_sobel_h;
                        cv::Mat yellow_sobel_vh;
                        
                        //cv::cvtColor(frame,yellow_sobel_v,CV_BGR2GRAY);
                        //cv::Sobel(yellow_hsv,yellow_sobel_v,CV_8UC1,1,0);
                        //if(!debug)cv::imshow("sobeltest_v",yellow_sobel_v);
                        // cv::Sobel(yellow_hsv,yellow_sobel_h,CV_8UC1,0,1);
                        // cv::Sobel(yellow_hsv,yellow_sobel_vh,CV_8UC1,0,1);
                        // if(debug)cv::imshow("sobeltest_h",yellow_sobel_h);
                        // if(debug)cv::imshow("sobeltest_vh",yellow_sobel_vh);
                        
                        //test_sobel
                        cv::cvtColor(frame,yellow_sobel_v,CV_BGR2GRAY);
                        cv::Sobel(yellow_sobel_v,yellow_sobel_v,yellow_sobel_v.depth(),1,0);
                        if(!debug)cv::imshow("sobeltest_v",yellow_sobel_v);
                        cv::Sobel(yellow_sobel_v,yellow_sobel_h,yellow_sobel_v.depth(),0,1);
                        if(!debug)cv::imshow("sobeltest_h",yellow_sobel_h);
                        //if(debug)cv::imshow("sobeltest_vh",yellow_sobel_vh);
                        
                        ////*Testing histogram*////
                        setProjection(callane, yellow_hsv, H_yResultWhite);
                                
                        ////*Detect lane by candidate label*////
                        std::vector<cv::Point> left_lane_fitting, right_lane_fitting; //이 곡률에 offset을 주면 가운데 선을 유지할 수 있겠지
                        cv::Point tmp_pt, left_mid = cv::Point(-1,-1), right_mid = cv::Point(-1,-1);
                        cv::Mat rect_result = frame.clone();
                        //cv::circle(rect_result,cv::Point(rect_result.cols/2,rect_result.rows-30),3,cv::Scalar(20,20,244),2);
                        cv::line(rect_result,center_pt_t,center_pt_b,cv::Scalar(50,20,233),2);
                        box_pt_y = callane.makeContoursLeftLane(yellow_hsv, yellow_labeling);//source img channel should be 1
                        box_pt_w = callane.makeContoursRightLane(white_hsv, white_labeling);//source img channel should be 1
                        cv::Point left_roi_t(0,frame.rows/2);
                        cv::Point left_roi_b(frame.cols/2-1,frame.rows-1);
                        cv::Point right_roi_t(frame.cols/2+1,frame.rows/2);
                        cv::Point right_roi_b(frame.cols-1,frame.rows-1);
			//left_robot_wheel_point, right_robot
			//이거로 바퀴 위치 알아내고 이때의 js.position[0],[1]등과 비교함.
			//정확히 가운데있을 경우의 js.position[0],[1]을 알아내서 에러값으로 얼마넘길지 정함
			//양 차선을 통해 얻은 ang_Vel또한 어떻게 적용해볼지 생각. 일단 이거로 먼저해보고 위에거 써보기.
			//
                        //**If you use vector type variable, Please cheack your variable is not empty! 
                        bool left_point = false, right_point = false;
                        if(!box_pt_y.empty() || !box_pt_w.empty()){
                                //**left roi editted if that needed
                                if(!box_pt_y.empty() && !box_pt_w.empty()){
                                        if(box_pt_y[1].x > left_roi_b.x) box_pt_y[1].x = left_roi_b.x;
                                        if(box_pt_y[0].y < left_roi_t.y) box_pt_y[0].y = left_roi_t.y;
                                        if(box_pt_y[0].x > left_roi_t.x){
                                                //**invailid lane roi to default roi
                                                box_pt_y[0] = left_roi_t;
                                                box_pt_y[1] = left_roi_b;
                                        }
                                // }
                                // //**left roi editted if that needed
                                // if(!box_pt_w.empty()){
                                        if(box_pt_w[0].x < right_roi_t.x) box_pt_w[0].x = right_roi_t.x;
                                        if(box_pt_w[0].y < right_roi_t.y) box_pt_w[0].y = right_roi_t.y;
                                        if(box_pt_w[1].x < right_roi_t.x){
                                                //**invailid lane roi to default roi
                                                box_pt_w[0] = right_roi_t;
                                                box_pt_w[1] = right_roi_b;
                                        }
                                }
                                
                                if(!box_pt_w.empty() && !box_pt_y.empty()){
                                        if(box_pt_y[0].x < box_pt_w[1].x){//두 차선을 모두 찾았고 박싱 위치가 올바를때의 조건
                                                //모든 차선이 무효일 경우에 대한 주행 알고리즘 작성 필요.(흰색 차선 찾기)
                                                if(abs(box_pt_y[1].y - box_pt_y[0].y) > 10){
                                                        if(for_gui) cv::rectangle(rect_result,box_pt_y[0],box_pt_y[1],cv::Scalar(0,0,255),2);
                                                        is_left_box_true = true;
                                                }
                                                if(abs(box_pt_w[1].y - box_pt_w[0].y) > 10){
                                                        if(for_gui) cv::rectangle(rect_result,box_pt_w[0],box_pt_w[1],cv::Scalar(0,0,255),2);
                                                        is_right_box_true = true;
                                                }
                                        }
                                        else{//두 차선을 모두 찾았지만 박싱 위치가 올바르지 않다면 아래쪽 박스를 선택
                                                if(box_pt_y[1].y > box_pt_w[1].y){
                                                        if(for_gui) cv::rectangle(rect_result,box_pt_y[0],box_pt_y[1],cv::Scalar(0,0,255),2);
                                                        is_left_box_true = true;
                                                }
                                                else{
                                                        if(for_gui) cv::rectangle(rect_result,box_pt_w[0],box_pt_w[1],cv::Scalar(0,0,255),2);
                                                        is_right_box_true = true;
                                                }
                                        }
                                }
                                else if(box_pt_w.empty()){//한 차선만 찾은 경우
                                        if(for_gui) cv::rectangle(rect_result,box_pt_y[0],box_pt_y[1],cv::Scalar(0,0,255),2);
                                        //right_interval = -9999;
                                        is_left_box_true = true;
                                }
                                else if(box_pt_y.empty()){
                                        if(for_gui) cv::rectangle(rect_result,box_pt_w[0],box_pt_w[1],cv::Scalar(0,0,255),2);
                                        //left_interval = -9999;
                                        is_right_box_true = true;
                                }
                                
                                center_pt_t = cv::Point(frame_width/2,0);
                                center_pt_b = cv::Point(frame_width/2,frame_height-1);
                                
                                //*for inner left lane fitting*//
                                cv::Point left_nodal_pt_t, left_nodal_pt_b;
                                double left_y_intercept;
                                double left_gradient;
                               
                                if(is_left_box_true && !box_pt_y.empty()){//여기선 empty조건은 뺴도 될
                                        for(int y = box_pt_y[1].y; y > box_pt_y[0].y; y--) {
                                                uchar* origin_data = yellow_labeling.ptr<uchar>(y);
                                                for(int x = box_pt_y[1].x; x > box_pt_y[0].x; x--) {                                
                                                        if(origin_data[x]!= (uchar)0) {
                                                                left_lane_fitting.push_back(cv::Point(x,y));
                                                                //cv::circle(rect_result,cv::Point(x,y),10,cv::Scalar(200,40,40));
                                                                break;
                                                        }
                                                }
                                                if(!left_lane_fitting.empty() && y == box_pt_y[0].y + 1){
                                                        //left_mid = cv::Point(left_lane_fitting[left_lane_fitting.size()/2].x,left_lane_fitting[left_lane_fitting.size()/2].y);
                                                        //cv::circle(rect_result,left_lane_fitting[10],10,cv::Scalar(200,40,40));
                                                        left_interval = center_pt_b.x - left_lane_fitting[10].x;
                                                        std::cout<<"left_interval : "<<center_pt_b.x - left_lane_fitting[10].x<<std::endl;
                                                        if(for_gui){
                                                                cv::line(rect_result,left_lane_fitting[10],left_lane_fitting[(left_lane_fitting.size()/4)*3],cv::Scalar(100,50,250),2);
                                                                // for(int i = 0; i<20; i++)
                                                                //         std::cout<<"left fit["<<i<<"] : "<<left_lane_fitting[i]<<std::endl;
                                                                left_nodal_pt_b = left_lane_fitting[10];
                                                                left_nodal_pt_t = left_lane_fitting[(left_lane_fitting.size()/4)*3];
                                                                if(left_nodal_pt_t.x - left_nodal_pt_b.x != 0){
                                                                        left_gradient = (left_nodal_pt_t.y - left_nodal_pt_b.y)/(left_nodal_pt_t.x - left_nodal_pt_b.x);
                                                                }
                                                                else{
                                                                        left_gradient = 9999;//for gradient setting infinity value
                                                                }
                                                                left_y_intercept = -left_gradient*(left_nodal_pt_b.x) - left_nodal_pt_b.y;////y = left_gradient*x + left_y_intercept
                                                                
                                                                //** left_lane_fitting equation: y = left_gradient*x + left_y_intercept
                                                                //** nodal point y (x is zero) : y = left_y_intercept
                                                                cv::line(rect_result,left_nodal_pt_b,center_pt_b,cv::Scalar(244,244,0),3);
                                                                left_theta = atan2f(abs((float)left_lane_fitting[10].y-(float)left_y_intercept), abs((float)left_lane_fitting[10].x - (float)center_pt_b.x));
                                                                std::cout<<"====left_theta :"<<left_theta<<std::endl;
                                                                
                                                                //** draw offset line(should be set offset x coordinate)
                                                                //cv::polylines(rect_result,left_lane_fitting,false,cv::Scalar(100,50,50),2);
                                                        }
                                                        
                                                        left_point = true;
                                                        break;
                                                }
                                        }
                                }///left///

                                
                                //*for inner rifght lane fitting*// 
                                cv::Point right_nodal_pt_t, right_nodal_pt_b;
                                double right_y_intercept;
                                double right_gradient;
                                //double right_theta, right_degree;
                                if(is_right_box_true && !box_pt_w.empty()){
                                        for(int y = box_pt_w[1].y; y>box_pt_w[0].y; y--) {
                                                uchar* origin_data = white_labeling.ptr<uchar>(y);
                                                for(int x = box_pt_w[0].x; x<box_pt_w[1].x; x++) {                                
                                                        if(origin_data[x]!= (uchar)0) {
                                                                right_lane_fitting.push_back(cv::Point(x,y));
                                                                break;
                                                        }
                                                }
                                                if(!right_lane_fitting.empty() && y == box_pt_w[0].y + 1){//박스안에 차선으로 꽉차있으면 이 조건에 안걸릴 수있으므로 조건 추가
                                                        //right_mid = cv::Point(right_lane_fitting[right_lane_fitting.size()/2].x,right_lane_fitting[right_lane_fitting.size()/2].y);
                                                        right_interval = right_lane_fitting[10].x - center_pt_b.x;
                                                        std::cout<<"right_interval : "<<right_lane_fitting[10].x - center_pt_b.x<<std::endl;
                                                        if(for_gui){
                                                                cv::line(rect_result,right_lane_fitting[10],right_lane_fitting[(right_lane_fitting.size()/4)*3],cv::Scalar(100,50,250),2);
                                                                right_nodal_pt_b = right_lane_fitting[10];
                                                                right_nodal_pt_t = right_lane_fitting[(right_lane_fitting.size()/4)*3];
                                                                if(right_nodal_pt_t.x - right_nodal_pt_b.x != 0){       
                                                                        right_gradient = (right_nodal_pt_t.y - right_nodal_pt_b.y)/(right_nodal_pt_t.x - right_nodal_pt_b.x);
                                                                }
                                                                else{
                                                                        right_gradient = 9999;//for gradient setting infinity value
                                                                }
                                                                right_y_intercept = -right_gradient*(right_nodal_pt_b.x) - right_nodal_pt_b.y;
                                                                //** right_lane_fitting equation : y = right_gradient*x + right_y_intercept **//
                                                                //** nodal point y (x is zero) : y = right_y_intercept **//
                                                                cv::line(rect_result,right_lane_fitting[10],center_pt_b,cv::Scalar(244,244,0),3);
                                                                right_theta = atan2f(abs((float)right_lane_fitting[10].y-(float)right_y_intercept),abs((float)right_lane_fitting[10].x - (float)center_pt_b.x));
                                                                std::cout<<"====right_theta :"<<right_theta<<std::endl;
                                                                //** draw offset line(should be set offset x coordinate) **//
                                                                //cv::polylines(rect_result,right_lane_fitting,false,cv::Scalar(100,50,50),2);
                                                        }
                                                        
                                                        right_point = true;
                                                        break;
                                                }
                                        }
                                }/// right////////////////////


                                ///이 정보는 보류...
                                // if( (is_left_box_true||is_right_box_true) && (left_point || right_point)){
                                //         if(left_point && right_point){// 이미 넘어온 영상에서 윗부분 거르니까 y값의 범위는 신경 안씀. x도 위에서 어느정도 보정됨
                                //                 try{
                                //                         if(left_mid.x>0 && left_mid.x<rect_result.cols-1){
                                //                                 cv::line(rect_result,cv::Point(left_mid.x,0),cv::Point(left_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2);
                                //                         }
                                //                         else if(left_mid.x<=0){
                                //                                 cv::line(rect_result,cv::Point(left_mid.x+1,0),cv::Point(left_mid.x+1,rect_result.rows-1),cv::Scalar(140,30,55),2);
                                //                         }
                                //                         else if(left_mid.x >= rect_result.cols-1){
                                //                                 cv::line(rect_result,cv::Point(left_mid.x-1,0),cv::Point(left_mid.x-1,rect_result.rows-1),cv::Scalar(140,30,55),2);
                                //                         }
                                //                         if(right_mid.x>0 && right_mid.x<rect_result.cols-1){
                                //                                 cv::line(rect_result,cv::Point(right_mid.x,0),cv::Point(right_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2); 
                                //                         }
                                //                         else if(right_mid.x <= 0){
                                //                                 cv::line(rect_result,cv::Point(right_mid.x,0),cv::Point(right_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2); 
                                //                         }
                                //                         else if(right_mid.x >= rect_result.cols-1){
                                //                                 cv::line(rect_result,cv::Point(right_mid.x,0),cv::Point(right_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2); 
                                //                         }
                                //                         cv::line(rect_result,cv::Point(right_mid.x,0),cv::Point(right_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2); 
                                //                         //cv::rectangle(rect_result,left_mid,right_mid,cv::Scalar(140,30,55),1);
                                //                 }
                                //                 catch(cv_bridge::Exception& e) {
                                //                         ROS_ERROR("rectangle error : %s", e.what());
                                //                         return;
                                //                 }
                                                
                                //         }
                                //         else if(!left_point && right_point){//left_point false
                                //               cv::line(rect_result,cv::Point(right_mid.x,0),cv::Point(right_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2);
                                //               //영상의 범위를 넘지 않도록 선 그릴
                                //         }
                                //         else if(!right_point && left_point){
                                //               cv::line(rect_result,cv::Point(left_mid.x,0),cv::Point(left_mid.x,rect_result.rows-1),cv::Scalar(140,30,55),2);
                                              
                                //         }
                                        
                                // }
                                
                                

                                ///////param goal test/////
                                // bool param_state;
                                // nh.setParam("/control/wait",true);
                                // nh.getParam("/control/wait",param_state);
                                // setPixelGoal(test,test_num);
                        
                                
                        }

                        ///** please make function
                        //계수들 다 런치파일로 옮기기
                        x_goal_ = 0.05; //default x goal
                        y_goal_ = 0; //default y goal

                        if(is_left_box_true && !is_right_box_true){//tracking left lane(yellow lane)
                                if(left_interval > left_min_interval && left_interval < left_max_interval){//go straight condition
                                        y_goal_ = 0;        
                                }
                                
                                        //** left lane tracking condition
                                        if(left_interval <= left_min_interval){//need right rotation(ang_vel<0 : right rotation)
                                                y_goal_ = -0.01;
                                                //y_goal_ = abs(left_theta);
                                        }
                                        else{//need left rotation(ang_vel>0 : left rotation)
                                                y_goal_ = 0.01;
                                                
                                        }

                                
                        }
                        else if(is_right_box_true && !is_left_box_true){//tracking right lane(white lane)
                                float right_degree = right_theta*180/CV_PI;
                                float left_degree = left_theta*180/CV_PI;
                                float sum_degree = right_degree + left_degree;
                                if(right_interval > right_min_interval && right_interval < right_max_interval){//go straight condition
                                        
                                        y_goal_ = 0;        
                                }
          
                                        //** right lane tracking condition
                                        if(right_interval <= right_min_interval){//need left rotation(ang_vel>0 : left rotation)
                                                y_goal_ = 0.01;
                                                
                                        }
                                        else{//need right rotation(ang_vel<0 : right rotation)
                                                y_goal_ = -0.01;
                                                
                                        }

                        }
                        else if(is_left_box_true && is_right_box_true){

                                //if all lane(yellow and white) detected, turtlebot following white lane path tracking
                                if(right_interval > right_min_interval && right_interval < right_max_interval){//go straight condition
                                        y_goal_ = 0;        
                                }
                                else{
                                        if(right_interval <= right_min_interval){//need left rotation(ang_vel>0 : left rotation)
                                                
                                                y_goal_ = 0.01;
                                        }
                                        else{//need right rotation(ang_vel<0 : right rotation)
                                                
                                                y_goal_ = -0.01;
                                        }
                                }
                                //x_goal_ = 0.1;//직진주행시 가속
                                if(left_interval > left_min_interval && left_interval < left_max_interval){//go straight condition
                                        y_goal_ = 0;        
                                }
                                //else if(left_interval + 20 > 50 && left_interval - 20 < 60){//중앙보다 차선쪽에 가까운경우
                                        //** left lane tracking condition
                                        if(left_interval <= left_min_interval){//need right rotation(ang_vel<0 : right rotation)
                                                y_goal_ = -0.01;
                                                //y_goal_ = abs(left_theta);
                                        }
                                        else{//need left rotation(ang_vel>0 : left rotation)
                                                y_goal_ = 0.01;
                                                //y_goal_ = abs(left_theta);
                                        }

                        }
                        else{//if detected no lane, than go straight
                                if(prev_y_goal_ < 0){
                                        x_goal_ = 0;//.08;
                                        y_goal_ = 0.04;
                                }
                                else if(prev_y_goal_ > 0){
                                        x_goal_ = 0;//.08;
                                        y_goal_ = -0.04;
                                }
                                else{
                                        y_goal_ = 0;
                                }
                                
                                
                        }

                        goal_array.data.clear();
                        goal_array.data.resize(0);
                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                        goal_array.data.push_back(y_goal_);
                        prev_y_goal_ = y_goal_;
                        std::cout<<"linear_vel"<<x_goal_<<std::endl;
                        std::cout<<"ang_vel"<<y_goal_<<std::endl;
                      
                        if(for_gui){
                                cv::imshow("result_rect",rect_result);
                        }
                        
                        ////*Restore birdeyeview img to origin view*////
                        restoreImgWithLangeMerge(callane,frame,yellow_labeling,white_labeling,mergelane);

                        ////*Make lane infomation msg for translate scan data*////
                        extractLanePoint(origin,mergelane);

                        output_origin_for_copy = origin.clone();
                        is_left_box_true = false;
                        is_right_box_true = false;
                        left_point = false;
                        right_point = false;
                        left_interval = 0;
                        right_interval = 0;
                        left_theta = 0;
                        right_theta = 0;
                }
                else{//frame is empty
                        while(frame.empty()) {//for unplugged camera
                                cv_ptr = cv_bridge::toCvCopy(img_msg,sensor_msgs::image_encodings::BGR8);
                                frame = cv_ptr->image;
                        }
                        x_goal_ = 0;
                        y_goal_ = 0;
                }
        }
        catch(cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception : %s", e.what());
                return;
        }

        if(auto_shot){
                std::cout<<"Save screen shot"<<std::endl;
                cv::imwrite("/home/seuleee/Desktop/autorace_img_src/0721/sign_signal/"+to_string(imgNum)+".jpg", output_origin_for_copy);
                imgNum++;
        }
        int ckey = cv::waitKey(10);
        if(ckey == 27) exit(1);
        else if(ckey == 32){//For save using space key
                std::cout<<"Save screen shot"<<std::endl;
                cv::imwrite("/home/seuleee/Desktop/autorace_img_src/"+to_string(imgNum)+".jpg", output_origin_for_copy);
                imgNum++;
        }
}

void InitImgObjectforROS::initParam(){
        nh.param<int>("/"+groupName+"/lane_detection/debug", debug, 0);
        nh.param<int>("/"+groupName+"/lane_detection/web_cam", web_cam, 0);
        nh.param<int>("/"+groupName+"/lane_detection/imshow", imshow, 0);
        nh.param<int>("/"+groupName+"/lane_detection/track_bar", track_bar, 0);
        nh.param<int>("/"+groupName+"/lane_detection/time_check", time_check, 0);
        nh.param<int>("/"+groupName+"/lane_detection/lable", lable, 0);
        nh.param<int>("/"+groupName+"/lane_detection/gazebo", gazebo, 1);
        nh.param<int>("/"+groupName+"/lane_detection/bird_eye_view", bird_eye_view, 0);
        nh.param<int>("/"+groupName+"/lane_detection/auto_shot", auto_shot, 0);
        nh.param<int>("/"+groupName+"/lane_detection/auto_record", auto_record, 0);
        
        nh.param<int>("/"+groupName+"/lane_detection/left_min_interval",left_min_interval,110);
        nh.param<int>("/"+groupName+"/lane_detection/left_max_interval",left_max_interval,150);
        nh.param<int>("/"+groupName+"/lane_detection/right_min_interval",right_min_interval,120);
        nh.param<int>("/"+groupName+"/lane_detection/right_max_interval",right_max_interval,160);
        nh.param<float>("/"+groupName+"/lane_detection/right_rotation_y_goal",right_rotation_y_goal,-0.1);
        nh.param<float>("/"+groupName+"/lane_detection/left_rotation_y_goal",left_rotation_y_goal,0.1);
        nh.param<float>("/"+groupName+"/lane_detection/default_x_goal",default_x_goal,0.05);
        nh.param<float>("/"+groupName+"/lane_detection/default_y_goal",default_y_goal,0);

        nh.param<int>("/"+groupName+"/lane_detection/y_hmin",y_hmin,15);
        nh.param<int>("/"+groupName+"/lane_detection/y_hmax",y_hmax,21);
        nh.param<int>("/"+groupName+"/lane_detection/y_smin",y_smin,52);
        nh.param<int>("/"+groupName+"/lane_detection/y_smax",y_smax,151);
        nh.param<int>("/"+groupName+"/lane_detection/y_vmin",y_vmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/y_vmax",y_vmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/w_hmin",w_hmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/w_hmax",w_hmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/w_smin",w_smin,0);
        nh.param<int>("/"+groupName+"/lane_detection/w_smax",w_smax,24);
        nh.param<int>("/"+groupName+"/lane_detection/w_vmin",w_vmin,172);
        nh.param<int>("/"+groupName+"/lane_detection/w_vmax",w_vmax,255);
        ROS_INFO("lane_detection %d %d %d %d %d %d %d", debug, web_cam, imshow, track_bar, time_check, lable, gazebo);
        ROS_INFO("imshow %d", imshow);
}

void InitImgObjectforROS::initMyHSVTrackbar(const string &trackbar_name){
                
                cv::namedWindow(trackbar_name, cv::WINDOW_AUTOSIZE);
                if (trackbar_name.find("YELLOW") != string::npos) { 

                        cv::createTrackbar("h min", trackbar_name, &y_hmin, 179, NULL);
                        cv::setTrackbarPos("h min", trackbar_name, y_hmin);

                        cv::createTrackbar("h max", trackbar_name, &y_hmax, 179, NULL);
                        cv::setTrackbarPos("h max", trackbar_name, y_hmax);

                        cv::createTrackbar("s min", trackbar_name, &y_smin, 255, NULL);
                        cv::setTrackbarPos("s min", trackbar_name, y_smin);

                        cv::createTrackbar("s max", trackbar_name, &y_smax, 255, NULL);
                        cv::setTrackbarPos("s max", trackbar_name, y_smax);

                        cv::createTrackbar("v min", trackbar_name, &y_vmin, 255, NULL);
                        cv::setTrackbarPos("v min", trackbar_name, y_vmin);

                        cv::createTrackbar("v max", trackbar_name, &y_vmax, 255, NULL);
                        cv::setTrackbarPos("v max", trackbar_name, y_vmax);
                }
                else if(trackbar_name.find("WHITE") != string::npos){

                        cv::createTrackbar("h min", trackbar_name, &w_hmin, 179, NULL);
                        cv::setTrackbarPos("h min", trackbar_name, w_hmin);

                        cv::createTrackbar("h max", trackbar_name, &w_hmax, 179, NULL);
                        cv::setTrackbarPos("h max", trackbar_name, w_hmax);

                        cv::createTrackbar("s min", trackbar_name, &w_smin, 255, NULL);
                        cv::setTrackbarPos("s min", trackbar_name, w_smin);

                        cv::createTrackbar("s max", trackbar_name, &w_smax, 255, NULL);
                        cv::setTrackbarPos("s max", trackbar_name, w_smax);

                        cv::createTrackbar("v min", trackbar_name, &w_vmin, 255, NULL);
                        cv::setTrackbarPos("v min", trackbar_name, w_vmin);

                        cv::createTrackbar("v max", trackbar_name, &w_vmax, 255, NULL);
                        cv::setTrackbarPos("v max", trackbar_name, w_vmax);
                }
}

void InitImgObjectforROS::setMyHSVTrackbarValue(const string &trackbar_name){

                if (trackbar_name.find("YELLOW") != string::npos) { 
                        y_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        y_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        y_smin = cv::getTrackbarPos("s min", trackbar_name);
                        y_smax = cv::getTrackbarPos("s max", trackbar_name);
                        y_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        y_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }
                else if(trackbar_name.find("WHITE") != string::npos){
                        w_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        w_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        w_smin = cv::getTrackbarPos("s min", trackbar_name);
                        w_smax = cv::getTrackbarPos("s max", trackbar_name);
                        w_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        w_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }

                nh.setParam("/"+groupName+"/lane_detection/y_hmin",y_hmin);
                nh.setParam("/"+groupName+"/lane_detection/y_hmax",y_hmax);
                nh.setParam("/"+groupName+"/lane_detection/y_smin",y_smin);
                nh.setParam("/"+groupName+"/lane_detection/y_smax",y_smax);
                nh.setParam("/"+groupName+"/lane_detection/y_vmin",y_vmin);
                nh.setParam("/"+groupName+"/lane_detection/y_vmax",y_vmax);

                nh.setParam("/"+groupName+"/lane_detection/w_hmin",w_hmin);
                nh.setParam("/"+groupName+"/lane_detection/w_hmax",w_hmax);
                nh.setParam("/"+groupName+"/lane_detection/w_smin",w_smin);
                nh.setParam("/"+groupName+"/lane_detection/w_smax",w_smax);
                nh.setParam("/"+groupName+"/lane_detection/w_vmin",w_vmin);
                nh.setParam("/"+groupName+"/lane_detection/w_vmax",w_vmax);
             
}

void InitImgObjectforROS::setColorPreocessing(lane_detect_algo::CalLane callane, cv::Mat src, cv::Mat& dst_y, cv::Mat& dst_w){
                ////*Make trackbar obj*////
                if(track_bar){
                        setMyHSVTrackbarValue(groupName+"_YELLOW_TRACKBAR");
                        setMyHSVTrackbarValue(groupName+"_WHITE_TRACKBAR");
                }

                ////*Make birdeyeview img*////
                cv::Mat bev;
                bev = src.clone();
                //cv::imshow(groupName+"bev",bev);
                if(groupName == "main"){//If you test by video, use one camera (please comment out the other camera)
                        if(bird_eye_view){
                                callane.birdEyeView_left(src,bev); //comment out by the other cam
                                if(debug) cv::imshow("bev_le",bev); //comment out by the other cam
                                //callane.birdEyeView_right(src,bev);
                                //if(debug) cv::imshow("bev_ri",bev);
                        }
                        else{
                                bev = src.clone();
                        }
                        
                }
                else if(groupName == "left"){
                        callane.birdEyeView_left(src,bev);
                        if(!debug) cv::imshow("bev_le",bev);
                }
                else if(groupName == "right"){
                        callane.birdEyeView_right(src,bev);
                        if(!debug) cv::imshow("bev_ri",bev);
                }
                
                
                ////*Detect yellow and white colors and make dst img to binary img by hsv value*////
                if (track_bar) {//Use trackbar. Use real-time trackbar's hsv value
                        callane.detectYHSVcolor(bev, dst_y, y_hmin, y_hmax, y_smin, y_smax, y_vmin, y_vmax);
                        callane.detectWhiteRange(bev,dst_w, w_hmin, w_hmax, w_smin, w_smax, w_vmin, w_vmax,0,0);
                        cv::imshow(groupName+"_YELLOW_TRACKBAR",dst_y);
                        cv::imshow(groupName+"_WHITE_TRACKBAR",dst_w);
                        
                }
                else {//Don't use trackbar. Use defalut value.
                        callane.detectYHSVcolor(bev, dst_y, 7, 21, 52, 151, 0, 180);
                        callane.detectWhiteRange(bev, dst_w, 0, 180, 0, 29, 179, 255,0,0);
                }       
}

void InitImgObjectforROS::setProjection(lane_detect_algo::CalLane callane, cv::Mat src, unsigned int* H_aix_Result_color){
                ////*Testing histogram*////
                cv::Mat histsrc = src.clone();
                cv::Mat dst = cv::Mat::zeros(histsrc.rows,histsrc.cols,CV_8UC1);
                callane.myProjection(histsrc,dst,H_aix_Result_color);
}

void InitImgObjectforROS::restoreImgWithLangeMerge(lane_detect_algo::CalLane callane, cv::Mat origin_size_img, cv::Mat src_y, cv::Mat src_w, cv::Mat& dst){
                ////*Restore birdeyeview img to origin view*////
                cv::Mat laneColor = src_y | src_w;
                dst = origin_size_img.clone();
                
                if(groupName == "left"){
                        callane.inverseBirdEyeView_left(laneColor, dst);
                }
                else if(groupName == "right"){
                        callane.inverseBirdEyeView_right(laneColor, dst); 
                }
                else if(groupName == "main"){//If you test by video, use one camera (please comment out the other camera)
                        if(bird_eye_view){
                                callane.inverseBirdEyeView_left(laneColor, dst); //comment out by the other cam
                                //callane.inverseBirdEyeView_right(laneColor, dst); 
                        }
                        else{
                                dst = laneColor.clone();
                        }        
                }
}

void InitImgObjectforROS::extractLanePoint(cv::Mat origin_src, cv::Mat lane_src){
                ////*Make lane infomation msg for translate scan data*////
                cv::Mat output_origin = origin_src.clone();
                cv::Mat pub_img = lane_src.clone();
                int coordi_count = 0;
                coordi_array.data.clear();
                coordi_array.data.push_back(10);
                for(int y = output_origin.rows-1; y>=0; y--) {
                        uchar* origin_data = output_origin.ptr<uchar>(y);
                        uchar* pub_img_data;
                        if(!gazebo){//for resizing prossesing img to webcam original image(640x320)
                                pub_img_data = pub_img.ptr<uchar>(y*0.5);//Restore resize img(0.5 -> 1))
                        }
                        else{//use gazebo topic(320x240) - None resize
                                pub_img_data = pub_img.ptr<uchar>(y); 
                        }
                        for(int x = 0; x<output_origin.cols; x++) {
                                int temp;
                                if(!gazebo){//for resizing prossesing img to webcam original image(640x320)
                                        temp = x*0.5; //Restore resize img(0.5 -> 1)
                                }
                                else{//use gazebo topic(320x240) - None resize
                                        temp = x;
                                }
                                if(pub_img_data[temp]!= (uchar)0) {
                                        coordi_count++;
                                        coordi_array.data.push_back(x);
                                        coordi_array.data.push_back(y);
                                        //origin_data[x*output_origin.channels()] = 255;
                                        origin_data[x*output_origin.channels()+1] = 25;
                                }
                        }
                }
                coordi_array.data[0] = coordi_count;

                ////*Result img marked lane*////
                cv::imshow(groupName+"_colorfulLane",output_origin);
}


 void InitImgObjectforROS::setPixelGoal(double* goal, int num){
        //  goal_array.data.clear();
        //  goal_array.data.push_back(goal[num]);
        //  goal_array.data.push_back(goal[num+1]);
        //  std::cout<<"==========goal visible :"<<goal_array.data[0]<<", "<<goal_array.data[1]<<std::endl;
        }

int main(int argc, char **argv){
        ros::init(argc, argv, "lane_detection");
        if(!gazebo){//if you subscribe topic that published camera_image pkg 
                groupName = argv[1];
        }
        else{//if you use another img topic
                groupName = "main";//for test pointxyzrgb to mat (/camera/depth_registerd/points) please set "light" or "right"
        }
        
        ROS_INFO("strat lane detection");
        InitImgObjectforROS img_obj;
        ros::Rate loop_rate(30);
        //record flag 만들기, group node launch파일에 복구
        if(auto_record){
                if(groupName == "left")
                        video_left.open("/home/seuleee/Desktop/autorace_video_src/0717/left_record.avi",cv::VideoWriter::fourcc('X','V','I','D'),fps,cv::Size(640/2,480/2), isColor);
                else if(groupName == "right")
                        video_right.open("/home/seuleee/Desktop/autorace_video_src/0717/right_record.avi",cv::VideoWriter::fourcc('X','V','I','D'),fps,cv::Size(640/2,480/2), isColor);
                else if(groupName == "main")
                        video_main.open("/home/seuleee/Desktop/autorace_video_src/0721/main_record.avi",cv::VideoWriter::fourcc('X','V','I','D'),fps,cv::Size(640/2,480/2), isColor);
        }
        
        while(img_obj.nh.ok()) {
                img_obj.pub.publish(img_obj.coordi_array);
                img_obj.ang_vel_pub.publish(img_obj.goal_array);
                img_obj.goal_pub.publish(img_obj.goal_array);
                ros::spinOnce();
                loop_rate.sleep();

        }
        ROS_INFO("program killed!\n");
        return 0;
}









///////not used////
void InitImgObjectforROS::initMyHSVTrackbar_old(const string &trackbar_name, int *hmin, int *hmax, int *smin, int *smax, int *vmin, int *vmax){
                cv::namedWindow(trackbar_name, cv::WINDOW_AUTOSIZE);

                cv::createTrackbar("h min", trackbar_name, hmin, 179, NULL);
                cv::setTrackbarPos("h min", trackbar_name, *(hmin));

                cv::createTrackbar("h max", trackbar_name, hmax, 179, NULL);
                cv::setTrackbarPos("h max", trackbar_name, *(hmax));

                cv::createTrackbar("s min", trackbar_name, smin, 255, NULL);
                cv::setTrackbarPos("s min", trackbar_name, *(smin));

                cv::createTrackbar("s max", trackbar_name, smax, 255, NULL);
                cv::setTrackbarPos("s max", trackbar_name, *(smax));

                cv::createTrackbar("v min", trackbar_name, vmin, 255, NULL);
                cv::setTrackbarPos("v min", trackbar_name, *(vmin));

                cv::createTrackbar("v max", trackbar_name, vmax, 255, NULL);
                cv::setTrackbarPos("v max", trackbar_name, *(vmax));

}

void InitImgObjectforROS::setMyHSVTrackbarValue_old(const string &trackbar_name,int *hmin, int *hmax, int *smin, int *smax, int *vmin, int *vmax){
                *hmin = cv::getTrackbarPos("h min", trackbar_name);
                *hmax = cv::getTrackbarPos("h max", trackbar_name);
                *smin = cv::getTrackbarPos("s min", trackbar_name);
                *smax = cv::getTrackbarPos("s max", trackbar_name);
                *vmin = cv::getTrackbarPos("v min", trackbar_name);
                *vmax = cv::getTrackbarPos("v max", trackbar_name);


                 if(groupName=="left"){
                        if(trackbar_name == "LEFT_YELLOW_TRACKBAR"){
                                nh.setParam("/"+groupName+"/lane_detection/y_hmin",y_hmin);
                                nh.setParam("/"+groupName+"/lane_detection/y_hmax",y_hmax);
                                nh.setParam("/"+groupName+"/lane_detection/y_smin",y_smin);
                                nh.setParam("/"+groupName+"/lane_detection/y_smax",y_smax);
                                nh.setParam("/"+groupName+"/lane_detection/y_vmin",y_vmin);
                                nh.setParam("/"+groupName+"/lane_detection/y_vmax",y_vmax);
                        }
                        if(trackbar_name == "LEFT_WHITE_TRACKBAR"){
                                nh.setParam("/"+groupName+"/lane_detection/w_hmin",w_hmin);
                                nh.setParam("/"+groupName+"/lane_detection/w_hmax",w_hmax);
                                nh.setParam("/"+groupName+"/lane_detection/w_smin",w_smin);
                                nh.setParam("/"+groupName+"/lane_detection/w_smax",w_smax);
                                nh.setParam("/"+groupName+"/lane_detection/w_vmin",w_vmin);
                                nh.setParam("/"+groupName+"/lane_detection/w_vmax",w_vmax);
                        }
                 }
                 else if(groupName == "right"){
                        if(trackbar_name == "RIGHT_YELLOW_TRACKBAR"){
                                nh.setParam("/"+groupName+"/lane_detection/y_hmin",y_hmin);
                                nh.setParam("/"+groupName+"/lane_detection/y_hmax",y_hmax);
                                nh.setParam("/"+groupName+"/lane_detection/y_smin",y_smin);
                                nh.setParam("/"+groupName+"/lane_detection/y_smax",y_smax);
                                nh.setParam("/"+groupName+"/lane_detection/y_vmin",y_vmin);
                                nh.setParam("/"+groupName+"/lane_detection/y_vmax",y_vmax);
                        }
                        if(trackbar_name == "RIGHT_WHITE_TRACKBAR"){
                                nh.setParam("/"+groupName+"/lane_detection/w_hmin",w_hmin);
                                nh.setParam("/"+groupName+"/lane_detection/w_hmax",w_hmax);
                                nh.setParam("/"+groupName+"/lane_detection/w_smin",w_smin);
                                nh.setParam("/"+groupName+"/lane_detection/w_smax",w_smax);
                                nh.setParam("/"+groupName+"/lane_detection/w_vmin",w_vmin);
                                nh.setParam("/"+groupName+"/lane_detection/w_vmax",w_vmax);
                        }
                 }
                 else if(groupName == "main"){
                         if(trackbar_name == "YELLOW_TRACKBAR"){
                                nh.setParam("/"+groupName+"/lane_detection/y_hmin",y_hmin);
                                nh.setParam("/"+groupName+"/lane_detection/y_hmax",y_hmax);
                                nh.setParam("/"+groupName+"/lane_detection/y_smin",y_smin);
                                nh.setParam("/"+groupName+"/lane_detection/y_smax",y_smax);
                                nh.setParam("/"+groupName+"/lane_detection/y_vmin",y_vmin);
                                nh.setParam("/"+groupName+"/lane_detection/y_vmax",y_vmax);
                        }
                        if(trackbar_name == "WHITE_TRACKBAR"){
                                nh.setParam("/"+groupName+"/lane_detection/w_hmin",w_hmin);
                                nh.setParam("/"+groupName+"/lane_detection/w_hmax",w_hmax);
                                nh.setParam("/"+groupName+"/lane_detection/w_smin",w_smin);
                                nh.setParam("/"+groupName+"/lane_detection/w_smax",w_smax);
                                nh.setParam("/"+groupName+"/lane_detection/w_vmin",w_vmin);
                                nh.setParam("/"+groupName+"/lane_detection/w_vmax",w_vmax);
                        }
                 }
}

