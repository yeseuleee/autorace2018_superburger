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
#include "std_msgs/Bool.h"
#include "std_msgs/Int8.h"
#include <time.h>
#include <string>
#include <math.h>

#define COORDI_COUNT 4000
#define CLOCK_PER_SEC 1000

/////////mission check///////
bool parking_mode = false, parking_checked = false;
int parking_reliabilty = 0, go_cnt = 0;
cv::Point first_point;
int parking_stage = 0; 
bool blocking_bar_mode = false, blocking_bar_checked = false;
int blocking_bar_reliabilty = 0, blocking_bar_stage = 0; 
bool tunnel_mode = false, tunnel_checked = false;
int tunnel_reliabilty = 0; 
bool signal_lamp_mode = false, signal_lamp_checked = false; 
int red_reliabilty = 0, green_reliabilty = 0, yellow_reliabilty = 0;
bool normal_mode = true;
bool curve_mode = false;
int curve_stage = 0;

int first_left = -5, first_right=-5, first_width = -1;
////////////////////////////////
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
//static int for_gui;
// 횡단보도 탐지방법 찾기
static const std::string record_name;

static int left_min_interval, left_max_interval;
static int right_min_interval, right_max_interval;
static float right_rotation_y_goal,left_rotation_y_goal;
static float default_x_goal, default_y_goal;

//** trackbar vailable **//
//for lane_color
static int y_hmin, y_hmax, y_smin, y_smax, y_vmin, y_vmax;
static int w_hmin, w_hmax, w_smin, w_smax, w_vmin, w_vmax;

//for blocking lamp
static int r_hmin, r_hmax, r_smin, r_smax, r_vmin, r_vmax;

//for signal lamp
static int r2_hmin, r2_hmax, r2_smin, r2_smax, r2_vmin, r2_vmax;
static int g_hmin, g_hmax, g_smin, g_smax, g_vmin, g_vmax;
static int y2_hmin, y2_hmax, y2_smin, y2_smax, y2_vmin, y2_vmax;

static int reset_msg;

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
double fps = 6;
int control_first = 0;
int fourcc = CV_FOURCC('X','V','I','D'); // codec
bool isColor = true;
cv::Point center_pt_t, center_pt_b;
int center_cali;

cv::VideoWriter video_left;
cv::VideoWriter video_right;
cv::VideoWriter video_main;

static std::string groupName;

lane_detect_algo::vec_mat_t lane_m_vec;
int dot_cnt = 0;

int empty_cnt = 0; //for detect tunnel
int pre_left_interval = -2, pre_right_interval = -2;
int left_interval = -1, right_interval = -1;
int left_interval_sum = 0, right_interval_sum = 0;
int pre_right_size = -1, pre_left_size = -1;
std::vector<int> left_interval_vec, right_interval_vec;
std::vector<int> left_direction_vec, right_direction_vec;
std::vector<int> left_size_vec, right_size_vec;

double left_slope = -999, right_slope = -999;
float left_ang_vel, right_ang_vel;
float x_goal_, y_goal_, prev_y_goal_;
float left_theta = 0;
float right_theta = 0;

std::vector<int> right_doubling_vec, left_doubling_vec;
int left_steer_avg = 999, right_steer_avg = 999;
int pre_left_steer_avg = 999, pre_right_steer_avg = 999;
int dynamic_center_x = 0, dynamic_center_y = 0;
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
        std_msgs::Bool reset_val;
        ros::Publisher pub = nh.advertise<std_msgs::Int32MultiArray>("/"+groupName+"/lane",100);//Topic publishing at each camera
        ros::Publisher goal_pub = nh.advertise<std_msgs::Float32MultiArray>("/"+groupName+"/pixel_goal",100);
        ros::Publisher ang_vel_pub = nh.advertise<std_msgs::Float32MultiArray>("/"+groupName+"/angular_vel",100);
        ros::Publisher reset_msg_pub = nh.advertise<std_msgs::Bool>("/"+groupName+"/reset_msg",100);
        int imgNum = 0;//for saving img
        int msg_count_left = 0, msg_count_right = 0;
        int left_direction_cnt = 0, right_direction_cnt = 0;
        int leftlane_turn_value_num = 0, rightlane_turn_value_num = 0;
        std::vector<int> leftlane_turn_value_vec, rightlane_turn_value_vec;
        cv::Mat output_origin_for_copy;//for saving img
        InitImgObjectforROS();
        ~InitImgObjectforROS();
        void depthMessageCallback(const sensor_msgs::PointCloud2::ConstPtr& input);
        void imgCb(const sensor_msgs::ImageConstPtr& img_msg);
        void initParam();
        void initMyHSVTrackbar(const string &trackbar_name);
        void initMyRESETTrackbar(const string &trackbar_name);
        void setMyHSVTrackbarValue(const string &trackbar_name);
        void setMyRESETTrackbarValue(const string &trackbar_name);
        void setColorPreocessing(lane_detect_algo::CalLane callane, cv::Mat src, cv::Mat& dst_y, cv::Mat& dst_w, cv::Mat& dst_r, cv::Mat& dst_r2, cv::Mat& dst_g, cv::Mat& dst_y2);
        void setProjection(lane_detect_algo::CalLane callane, cv::Mat src, unsigned int* H_aix_Result_color);
        void restoreImgWithLangeMerge(lane_detect_algo::CalLane callane, cv::Mat origin_size_img, cv::Mat src_y, cv::Mat src_w, cv::Mat& dst);
        void extractLanePoint(cv::Mat origin_src, cv::Mat lane_src);
        void initMyHSVTrackbar_old(const string &trackbar_name, int *hmin, int *hmax, int *smin, int *smax, int *vmin, int *vmax);
        void setMyHSVTrackbarValue_old(const string &trackbar_name,int *hmin, int *hmax, int *smin, int *smax, int *vmin, int *vmax);
        void setPixelGoal(double* goal, int num);
        void setMySobelwithROI(cv::Mat src, double delete_row_per, const string &lane_name, cv::Mat& dst);
        void setMyLaneBox(cv::Point t_pt, cv::Point b_pt, const string &lane_name, std::vector<cv::Point> &dst);
        bool setMyLaneFitting(cv::Mat& src_img, std::vector<cv::Point> src_pt, const string &lane_name, std::vector<cv::Point> &dst);
        int FittedLaneCheck(std::vector<cv::Point> src, const string &lane_name);
        void setMyCannywithROI(const string &lane_name, cv::Mat &dst);
        void errorLaneCheck(const string &lane_name, cv::Mat& src, std::vector<cv::Point> &lane_data, std::vector<cv::Point> lane_roi);
        void errorLaneCheck2(cv::Mat& src, std::vector<cv::Point> &left_lane_data, std::vector<cv::Point> &right_lane_data);
        int slopeCheck(cv::Mat &src, std::vector<cv::Point> lane_data, const string &lane_name);
        void calSlope(cv::Mat &src, std::vector<cv::Point> lane_data, const string &lane_name);
        bool detectBlockingBar(cv::Mat src);
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
                initMyHSVTrackbar(groupName+"_YELLOW_LANE_TRACKBAR");
                initMyHSVTrackbar(groupName+"_WHITE_LANE_TRACKBAR");
                initMyHSVTrackbar(groupName+"_BLOCKING_RED_TRACKBAR");
                initMyHSVTrackbar(groupName+"_SIGNAL_RED_TRACKBAR");
                initMyHSVTrackbar(groupName+"_SIGNAL_GREEN_TRACKBAR");
                initMyHSVTrackbar(groupName+"_SIGNAL_YELLOW_TRACKBAR");
                initMyRESETTrackbar("reset msg");
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
        cv::Mat red_hsv, red2_hsv, green_hsv, yellow2_hsv;
        std::vector<cv::Point> box_pt_y,box_pt_w;
        std::vector<cv::Point> over_pt_y, over_pt_w;
        
        uint frame_height, frame_width;
        //bool is_left_box_true = false, is_right_box_true = false;        
        try{
                cv_ptr = cv_bridge::toCvCopy(img_msg,sensor_msgs::image_encodings::BGR8);
                frame = cv_ptr->image;
                origin = cv_ptr->image;
                //rec_img = cv_ptr->image;
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
                        cv::Mat gui_test = frame.clone();
                        lane_detect_algo::CalLane callane;


                        
                        ////*Testing histogram*////탑뷰 아니면 쓰기 힘들다..
                        // unsigned int* H_yResultYellow = new unsigned int[frame_width];
                        // std::memset(H_yResultYellow, 0, sizeof(uint) * frame_width);
                        // unsigned int* H_yResultWhite = new unsigned int[frame_width];
                        // std::memset(H_yResultWhite, 0, sizeof(uint) * frame_width);
                        // unsigned int* H_xResultYellow = new unsigned int[frame_height];
                        // std::memset(H_xResultYellow, 0, sizeof(uint) * frame_height);
                        // unsigned int* H_xResultWhite = new unsigned int[frame_height];
                        // std::memset(H_xResultWhite, 0, sizeof(uint) * frame_height);
                        //setProjection(callane, yellow_hsv, H_yResultWhite);
                        
                        ///////*reset trackbar set*//////////////////
                        
                        if(track_bar){
                                cv::Mat reset_img = cv::Mat::zeros(frame.size(), CV_8UC3);
                                setMyRESETTrackbarValue("reset msg");
                                cv::imshow("reset msg",reset_img);
                                reset_val.data = reset_msg;
                        }
                        
                        ////*Process color detection including trackbar setting*////
                        //컬러영상에 소벨엣지 적용해서 노란색 추출하는방법 생각해보기
                        
                        //setColorPreocessing(callane, sobel_v, yellow_hsv, white_hsv);
                        setColorPreocessing(callane, frame, yellow_hsv, white_hsv, red_hsv, red2_hsv, green_hsv, yellow2_hsv);
                        cv::Mat origin_white_hsv = white_hsv.clone();
                        cv::Mat origin_yellow_hsv = yellow_hsv.clone();
                        //cv::Mat yellow_sobel;
                        //setMySobelwithROI(frame,0.2,"left",yellow_sobel);
                        cv::Mat yellow_canny = frame.clone();
                        setMyCannywithROI("left",yellow_canny);//인자로 범위 받는걸로 고치기..
                        callane.makeContoursLeftLane(yellow_canny,yellow_canny);
                        //cv::imshow("canny_test",yellow_canny);
                        
                        //  cv::Mat white_canny = frame.clone();
                        //  setMyCannywithROI("right",white_canny);
                        //  callane.makeContoursRightLane(white_canny,white_canny);
                        // cv::imshow("white_canny",white_canny);
                        
                        if(!blocking_bar_checked){
                                blocking_bar_mode = detectBlockingBar(red_hsv);
                                if(blocking_bar_mode){
                                        normal_mode = false;
                                        go_cnt = 0;
                                        std::cout<<"#################################### detect blocking bar ##############"<<std::endl;
                                }
                        }
                        



                        cv::Mat white_sobel;
                        setMySobelwithROI(frame,0.2,"right",white_sobel);
                         
                        //cv::Mat gui_img = frame.clone();
                        cv::Point left_roi_t(10,frame.rows/2);
                        cv::Point left_roi_b(frame.cols/2,frame.rows-3);
                        cv::Point right_roi_t(frame.cols/2,frame.rows/2);
                        cv::Point right_roi_b(frame.cols-3,frame.rows-3);
                        

                        box_pt_y.push_back(left_roi_t);
                        box_pt_y.push_back(left_roi_b);
                        over_pt_y.push_back(cv::Point(left_roi_b.x - 10, left_roi_t.y));
                        over_pt_y.push_back(cv::Point(left_roi_b.x + 10, left_roi_b.y));
                        yellow_labeling = yellow_canny.clone();
                        
                        //yellow_labeling = yellow_sobel.clone();
                        int white_valid;
                        box_pt_w.push_back(right_roi_t);
                        box_pt_w.push_back(right_roi_b);
                        over_pt_w.push_back(cv::Point(right_roi_t.x - 10, right_roi_t.y));
                        over_pt_w.push_back(cv::Point(right_roi_t.x + 10, right_roi_b.y));
                        cv::Mat element(3,3,CV_8U,cv::Scalar(1)); 
                        cv::dilate(white_hsv,white_hsv,element);
                        cv::medianBlur(white_hsv, white_hsv, 1);
                        for(int y = white_hsv.rows-3; y<white_hsv.rows; y++){
                                uchar* none_roi_data = white_hsv.ptr<uchar>(y);
                                for(int x = 0; x<white_hsv.cols; x++){
                                        if(none_roi_data[x] != (uchar)0){
                                                none_roi_data[x] = (uchar)0;
                                        }
                                }
                        }
                        cv::Mat over_w_img = white_hsv.clone();
                        cv::Mat over_y_img = yellow_labeling.clone();
                        
                        white_valid = callane.makeContoursRightLane(white_hsv, white_labeling);
                        
                        

                        center_pt_t = cv::Point(frame_width/2,0);
                        center_pt_b = cv::Point(frame_width/2,frame_height-1);

                        //**If you use vector type variable, Please cheack your variable is not empty! 
                        //bool left_point = false, right_point = false;
                        std::vector<cv::Point> left_lane_fitting, right_lane_fitting, dot_lane_fitting;  
                        int left_lane_turn, right_lane_turn;//, dot_lane_turn;
                        //*for inner left lane fitting*//
                        
                        if(setMyLaneFitting(yellow_labeling, box_pt_y, "left", left_lane_fitting)){                  
                                left_lane_turn = FittedLaneCheck(left_lane_fitting,"left");
                                if(left_lane_turn != 3){
                                        leftlane_turn_value_num++;
                                }
                                leftlane_turn_value_vec.push_back(left_lane_turn);
                        }
                        else{
                                left_lane_fitting.clear();
                                left_lane_fitting.resize(0);
                                yellow_labeling = cv::Mat::zeros(yellow_labeling.size(), CV_8UC3);
                                if(debug) std::cout<<"invalid left lane"<<std::endl;
                        }
                         //int dot_c = 0;
                        if(white_valid != -1){
                                //*for inner right lane fitting*// 
                                if(setMyLaneFitting(white_labeling, box_pt_w, "right", right_lane_fitting)){    
                                        right_lane_turn = FittedLaneCheck(right_lane_fitting,"right");
                                        if(right_lane_turn != 3){
                                                rightlane_turn_value_num++;
                                        }
                                        rightlane_turn_value_vec.push_back(right_lane_turn);
                                }
                                //else{  
                                
                                //도트는 선 따로 피팅하자~~
                                // else{
                                //         right_lane_fitting.clear();
                                //         right_lane_fitting.resize(0);
                                //         white_labeling = cv::Mat::zeros(white_labeling.size(), CV_8UC3);  
                                //         if(debug) std::cout<<"invaild right label"<<std::endl;
                                // }
                              //  }
                        }       
                        
                        //cv::imshow("dot",gui_test);
                        //이방법이 잘 안되면 hsv and연산 , canny(label) and연산 추가해 비교하기
                        //오른쪽 차선과 연결된 영역이 왼쪽 차선 영역에 나타나면 오른쪽 차선의 마지막을 시작점으로해 그 윗부분을 날림(색깔 비교안넣고 그냥 날리는 버전)
                        //왼쪽 차선과 연결된 영역이 오른쪽 차선 영역에 나타나면 위와같이 날림(canny and연산 안하고 그냥 날리는 버전)
                        cv::Mat tt = frame.clone();
                        if(!right_lane_fitting.empty() && !left_lane_fitting.empty()){
                                
                                //left roi에 나타나는 흰색선 제거
                                int over_w_start_y = right_lane_fitting[right_lane_fitting.size()-1].y, over_w_end_y = 0;
                                int over_w_start_x = right_lane_fitting[right_lane_fitting.size()-1].x, over_w_end_x = 0;
                                
                                //delete left lane that area of white lane
                                if(abs(over_w_start_x - right_roi_t.x)<15){
                                        for(int y = over_w_start_y; y>over_w_end_y; y--){
                                                uchar* delete_data = yellow_labeling.ptr<uchar>(y);
                                                for(int x = over_w_start_x; x > over_w_end_x; x--){
                                                        if(delete_data[x] != (uchar)0){
                                                                delete_data[x] = (uchar)0;
                                                        }
                                                }
                                        }
                                }
                                int over_y_start_y = left_lane_fitting[left_lane_fitting.size()-1].y, over_y_end_y = 0;
                                int over_y_start_x = left_lane_fitting[left_lane_fitting.size()-1].x, over_y_end_x = white_labeling.cols;
                                //delete left lane that area of white lane
                                if(abs(over_y_start_x - left_roi_b.x)<15){
                                        for(uint y = over_y_start_y; y>over_y_end_y; y--){
                                                uchar* delete_data = white_labeling.ptr<uchar>(y);
                                                for(uint x = over_y_start_x; x < over_y_end_x; x++){
                                                        if(delete_data[x] != (uchar)0){
                                                                delete_data[x] = (uchar)0;
                                                        }
                                                }
                                        }
                                }

                                left_lane_fitting.clear();
                                left_lane_fitting.resize(0);
                                if(setMyLaneFitting(yellow_labeling, box_pt_y, "left", left_lane_fitting)){                  
                                        FittedLaneCheck(left_lane_fitting,"left");
                                }
                                else{
                                        if(!left_lane_fitting.empty()){
                                                left_lane_fitting.clear();
                                                left_lane_fitting.resize(0);
                                        }
                                        yellow_labeling = cv::Mat::zeros(yellow_labeling.size(), CV_8UC1);//1채널 이미지다!!!!!!주의하기 CV_8UC1이다!!!
                                        if(debug) std::cout<<"invalid left lane"<<std::endl;
                                }
                                right_lane_fitting.clear();
                                right_lane_fitting.resize(0);
                                if(setMyLaneFitting(white_labeling, box_pt_w, "right", right_lane_fitting)){    
                                        FittedLaneCheck(right_lane_fitting,"right");
                                }
                                else{
                                        if(right_lane_fitting.empty()){
                                                right_lane_fitting.clear();
                                                right_lane_fitting.resize(0);
                                        }
                                        white_labeling = cv::Mat::zeros(white_labeling.size(), CV_8UC1);  
                                }
                                
                               
                        }
                        cv::Mat slope_test = frame.clone();
                        // if(!left_lane_fitting.empty()){
                        //         slopeCheck(slope_test,left_lane_fitting,"left");
                        // }
                        // if(!right_lane_fitting.empty()){
                                
                        //         slopeCheck(slope_test,right_lane_fitting,"right");
                        // }
                        // cv::imshow("direction",slope_test);

                       // cv::polylines(gui_test,dot_lane_fitting,0,cv::Scalar(222,232,22),3);
                        
                        //cv::polylines(gui_test,left_lane_fitting,0,cv::Scalar(20,200,240),2);
                        //cv::polylines(gui_test,right_lane_fitting,0,cv::Scalar(200,100,23),2);

                        
                        cv::Point left_fit_pt = cv::Point(-1,-1);
                        //if(!left_lane_fitting.empty()){     
                                if(!left_lane_fitting.empty()){
                                        if(left_lane_fitting.size() > 10){
                                                for(int i = 0; i<left_lane_fitting.size(); i++){
                                                        if(left_lane_fitting[i].y >160 && left_lane_fitting[i].y<170){
                                                                left_fit_pt = cv::Point(left_lane_fitting[i].x, left_lane_fitting[i].y);
                                                                break;
                                                        }
                                                }
                                                int l_sum = 0, l_avg = 0, l_cnt =0;
                                                for(int i = 0; i<left_lane_fitting.size(); i++){
                                                        if(left_lane_fitting[i].y >160 && left_lane_fitting[i].y<170){
                                                               std::cout<<"left_ lane : "<< abs(left_lane_fitting[i].x - center_pt_b.x)<<std::endl;
                                                               l_sum += abs(left_lane_fitting[i].x - center_pt_b.x);
                                                               l_cnt++;
                                                        }
                                                }
                                                if(l_cnt != 0){
                                                        l_avg = l_sum/l_cnt;
                                                        std::cout<<"l_avg : "<<l_avg<<std::endl;
                                                }
                                                if(first_left < -2){
                                                        first_left++;
                                                }
                                                else if(first_left == -2){
                                                        left_interval = l_avg;//abs(left_fit_pt.x - center_pt_b.x);
                                                        if(left_interval < 10) left_interval = left_min_interval+1;
                                                        first_left = left_interval;
                                                        std::cout<<"####################first left set : "<<first_left<<std::endl;
                                                }
                                                left_interval = l_avg;//abs(left_fit_pt.x - center_pt_b.x);
                                                //calSlope(slope_test,left_lane_fitting,"left");
                                                // if(msg_count_left < 6){
                                                        
                                                //         //int left_dir = 0;
                                                //         //left_dir = slopeCheck(slope_test,left_lane_fitting,"left");
                                                //         //left_direction_vec.push_back(left_dir);
                                                //         left_size_vec.push_back(left_lane_fitting.size());
                                                //         // if(left_dir == -1){
                                                //         //         left_direction_cnt++;
                                                //         // } 
                                                //         //left_direction_vec.push_back(slopeCheck(slope_test,left_lane_fitting,"left"));
                                                //         if(left_fit_pt.x != -1){
                                                //                 int edit = 1;
                                                //                 if(pre_left_interval != -1 && left_interval != -1){
                                                //                         if(abs(left_interval - abs(left_fit_pt.x - center_pt_b.x)) > 9){
                                                //                                 left_interval_vec.push_back(abs(left_fit_pt.x - center_pt_b.x));
                                                //                                 left_interval_sum += abs(left_fit_pt.x - center_pt_b.x);
                                                //                                 edit = 0;
                                                //                         }
                                                //                 }
                                                //                 if(edit==1){
                                                //                         left_interval_vec.push_back(abs(left_fit_pt.x - center_pt_b.x));
                                                //                         left_interval_sum += abs(left_fit_pt.x - center_pt_b.x);
                                                //                 }
                                                //                 // left_interval_vec.push_back(abs(center_pt_b.x - left_fit_pt.x));
                                                //                 // left_interval_sum += abs(center_pt_b.x - left_fit_pt.x);  
                                                //         }
                                                //         else{
                                                //                 int tmp = -1;//left_min_interval +1;
                                                //                 left_interval_vec.push_back(tmp);
                                                //                 left_interval_sum += abs(tmp);
                                                //         }
                                                // }
                                                // else{
                                                //         int left_interval_avg = left_interval_sum/left_interval_vec.size();
                                                //         std::vector<int> avg_diff_vec;
                                                //         for(uint j = 0; j<left_interval_vec.size(); j++){
                                                //                 avg_diff_vec.push_back(abs(left_interval_vec[j]-left_interval_avg));
                                                //         }
                                                //         int max_diff = -9999;
                                                //         for(uint j = 0; j<avg_diff_vec.size(); j++){
                                                //                 if(avg_diff_vec[j] > max_diff){
                                                //                         max_diff = avg_diff_vec[j];
                                                //                 }
                                                //         }
                                                //         left_interval_sum = 0;
                                                //         int bigger_diff_sum = 0, lower_diff_sum = 0;
                                                //         int error_diff = 0, bigger_cnt = 0, lower_cnt = 0;
                                                //         for(uint j = 0; j<avg_diff_vec.size(); j++){
                                                //                 // if(abs(avg_diff_vec[j] - max_diff) < 10){
                                                //                 //         left_interval_sum += left_interval_vec[j];
                                                //                 //         sum_cnt++;
                                                //                 // }
                                                               
                                                //                 if(abs(avg_diff_vec[j] - max_diff) < 10){
                                                //                         bigger_diff_sum += left_interval_vec[j];
                                                //                         error_diff++;
                                                //                         bigger_cnt++;
                                                //                 }
                                                //                 else{
                                                //                         lower_diff_sum += left_interval_vec[j];
                                                //                         lower_cnt++;
                                                //                 }
                                                //         }
                                                //         if((uint)error_diff >= avg_diff_vec.size()/2){
                                                //                 if(bigger_cnt != 0) left_interval = bigger_diff_sum/bigger_cnt;
                                                //         }
                                                //         else{
                                                //                 if(lower_cnt != 0) left_interval = lower_diff_sum/lower_cnt;
                                                //         }
                                                        
                                                // }
                                               // msg_count_left++;
                                        }
                                        else{
                                                left_interval = -1;
                                               // msg_count_left = 0;
                                               // left_interval_vec.clear();
                                               // left_interval_vec.resize(0);
                                        }
                                        
                                }
                                else{
                                        left_interval = -1;
                                }
                        //}
                        
                        cv::Point right_fit_pt = cv::Point(-1,-1);
                        //if(!right_lane_fitting.empty()){
                                if(!right_lane_fitting.empty()){
                                       
                                        if(right_lane_fitting.size()>10){//사실 set lane fitting에서 이미 걸러져 나온 결과임 (20이상으로 걸러져나몽ㅁ)
                                               // calSlope(slope_test,right_lane_fitting,"right");
                                                for(int i = 0; i<right_lane_fitting.size(); i++){
                                                                if(right_lane_fitting[i].y >170 && right_lane_fitting[i].y<180){
                                                                        right_fit_pt = cv::Point(right_lane_fitting[i].x, right_lane_fitting[i].y);
                                                                        break;
                                                                }
                                                }
                                                int r_sum = 0, r_avg = 0, r_cnt =0;
                                                for(int i = 0; i<right_lane_fitting.size(); i++){
                                                        if(right_lane_fitting[i].y >170 && right_lane_fitting[i].y<180){
                                                               std::cout<<"right_ lane : "<<abs(right_fit_pt.x - center_pt_b.x)<<std::endl;
                                                               r_sum += abs(right_fit_pt.x - center_pt_b.x);
                                                               r_cnt++;
                                                        }
                                                }
                                                if(r_cnt != 0){
                                                        r_avg = r_sum/r_cnt;
                                                        std::cout<<"r_avg : "<<r_avg<<std::endl;
                                                }
                                                if(first_right < -2){
                                                        first_right++;
                                                }
                                                else if(first_right == -2){
                                                        right_interval = r_avg;//abs(right_fit_pt.x - center_pt_b.x);
                                                        first_right = right_interval;
                                                        if(right_interval <10) right_interval = right_min_interval+1;
                                                        std::cout<<"####################first right set : "<<first_right<<std::endl;
                                                }
                                                right_interval = r_avg;//abs(right_fit_pt.x - center_pt_b.x);
                                                // int right_dir = 0;
                                                // right_dir = slopeCheck(slope_test,right_lane_fitting,"right");
                                                // if(right_dir == -1){

                                                //         right_direction_cnt++;
                                                // }
                                                //int right_dir = 0;
                                                //right_dir = slopeCheck(slope_test,right_lane_fitting,"right");
                                                //right_direction_vec.push_back(right_dir);
                                                //right_size_vec.push_back(right_lane_fitting.size()); 
                                                // if(msg_count_right < 6){
                                                //         if(right_fit_pt.x != -1){
                                                //                 int edit = 1;
                                                //                 if(pre_right_interval != -1 && right_interval != -1){
                                                //                         if(abs(right_interval - abs(right_fit_pt.x - center_pt_b.x)) > 9){
                                                //                                 right_interval_vec.push_back(abs(right_fit_pt.x - center_pt_b.x));
                                                //                                 right_interval_sum += abs(right_fit_pt.x - center_pt_b.x);
                                                //                                 edit = 0;
                                                //                         }
                                                //                 }
                                                //                 if(edit==1){
                                                //                         right_interval_vec.push_back(abs(right_fit_pt.x - center_pt_b.x));
                                                //                         right_interval_sum += abs(right_fit_pt.x - center_pt_b.x);
                                                //                 }
                                                //         }
                                                //         else{
                                                //                 int tmp = -1;//right_min_interval + 1;
                                                //                 right_interval_vec.push_back(tmp);
                                                //                 right_interval_sum += tmp;
                                                //         }                                                
                                                         
                                                        
                                                // }
                                                // else{
                                                //         int right_interval_avg = right_interval_sum/right_interval_vec.size();
                                                //         std::vector<int> avg_diff_vec;
                                                //         for(uint j = 0; j<right_interval_vec.size(); j++){
                                                //                  avg_diff_vec.push_back(abs(right_interval_vec[j]-right_interval_avg));
                                                //         }
                                                //         int max_diff = -9999;
                                                //         for(uint j = 0; j<avg_diff_vec.size(); j++){
                                                //                 if(avg_diff_vec[j] > max_diff){
                                                //                          max_diff = avg_diff_vec[j];
                                                //                 }
                                                //         }
                                                //         right_interval_sum = 0;
                                                //         int bigger_diff_sum = 0, lower_diff_sum = 0;
                                                //         int error_diff = 0, bigger_cnt = 0, lower_cnt = 0;
                                                //         for(uint j = 0; j<avg_diff_vec.size(); j++){
                                                //                 if(abs(avg_diff_vec[j] - max_diff) < 10){
                                                //                         bigger_diff_sum += right_interval_vec[j];  
                                                //                         error_diff++;
                                                //                         bigger_cnt++;
                                                //                 }
                                                //                 else{
                                                //                         lower_diff_sum += right_interval_vec[j];
                                                //                         lower_cnt++;
                                                //                 }
                                                //         }   
                                                //         if((uint)error_diff >= avg_diff_vec.size()/2){
                                                //                 if(bigger_cnt != 0) right_interval = bigger_diff_sum/bigger_cnt;
                                                //         }
                                                //         else{
                                                //                 if(lower_cnt != 0) right_interval = lower_diff_sum/lower_cnt;
                                                //         }
                                                // }
                                              //  msg_count_right++;
                                        }
                                        else{
                                                right_interval = -1;
                                                // msg_count_right = 0;
                                                // right_interval_vec.clear();
                                               // right_interval_vec.resize(0);
                                        }
                                }
                                else{
                                        right_interval = -1;
                                }
                       // }
                        if(!left_lane_fitting.empty() && !right_lane_fitting.empty()){
                                first_width = first_left + first_right;
                                // left_direction_vec.push_back(slopeCheck(slope_test,left_lane_fitting,"left"));
                                // right_direction_vec.push_back(slopeCheck(slope_test,right_lane_fitting,"right"));
                                // if(left_direction_vec.size()>4 && right_direction_vec.size()>4){
                                //         int sum = 0;
                                //         for(int i = 0; i<4; i++){
                                //                 sum += left_direction_vec[i] + right_direction_vec[i];
                                //         }
                                //         if(sum == 0){
                                //                 left_interval = left_min_interval + 1;
                                //                 right_interval = right_min_interval + 1;
                                //         }       
                                // }
                                // left_direction_vec.clear();
                                // left_direction_vec.resize(0);
                                // right_direction_vec.clear();
                                // right_direction_vec.resize(0);
                        }
                        if(!parking_checked){
                                std::vector<cv::Point> dot_test_box_pt_w;
                                dot_test_box_pt_w.push_back(cv::Point(right_roi_t.x-50,right_roi_t.y));
                                dot_test_box_pt_w.push_back(right_roi_b);
                                
                                if(setMyLaneFitting(white_hsv,dot_test_box_pt_w, "dot_test", dot_lane_fitting)){
                                        
                                        if(dot_cnt != -1){
                                                dot_cnt++;
                                                std::cout<<"###################################################################### dot"<<std::endl;
                                        }
                                        if(dot_cnt > 3){ 
                                                dot_cnt = -1;
                                                parking_mode = true;
                                                normal_mode = false;
                                        }
                                        cv::polylines(gui_test,dot_lane_fitting,0,cv::Scalar(222,232,22),3);    
                                }
                                else{
                                        //if(dot_cnt != -1) dot_cnt = 0;
                                }
                        }
                        
                        ///////**param goal test/////
                        // bool param_state;
                        // nh.setParam("/control/wait",true);
                        // nh.getParam("/control/wait",param_state);
                        // setPixelGoal(test,test_num);
                        
                        //left와 right인터벌을 뽑아내는 벡터 위치를 보정할지 말지 결정하자.
                        if(normal_mode && !curve_mode){
                                x_goal_ = 0.08;
                                y_goal_ = 0.0;
                                std::cout<<"    right_lane_size : "<<right_lane_fitting.size()<<std::endl;
                                std::cout<<"    left_lane_size : "<<left_lane_fitting.size()<<std::endl;
                               // if(msg_count_left >= 7 ||  msg_count_right >= 7){
                                        
                                        if(pre_left_interval != -1 && left_interval != -1){
                                               // if(abs(pre_left_interval - left_interval) > 20) left_interval = -1;//left_min_interval + 1;
                                        }
                                        if(pre_right_interval != -1 && right_interval != -1){
                                                //if(abs(pre_right_interval - right_interval) > 20) right_interval = -1;//right_min_interval;
                                        }
                                        
                                        if(!left_lane_fitting.empty() && !right_lane_fitting.empty()){//tracking left lane(yellow lane)
                                                // if(abs(abs(left_interval + right_interval) - first_width) > 45){
                                                //         if(left_interval > 100 && right_interval > 100){
                                                //                 if(pre_left_interval > pre_right_interval) right_interval = -1;
                                                //                 else left_interval = -1;
                                                //         }
                                                        
                                                // }
                                                int left_size = left_lane_fitting.size(), right_size = right_lane_fitting.size();
                                                // int left_size_sum = 0, right_size_sum = 0;
                                                // if(!left_size_vec.empty() && !right_size_vec.empty()){
                                                //         for(int i = 0; i<left_size_vec.size(); i++){
                                                //                 left_size_sum += left_size_vec[i];
                                                //         }
                                                //         left_size = left_size_sum/left_size_vec.size();
                                                //         for(int i = 0; i<right_size_vec.size(); i++){
                                                //                 right_size_sum += right_size_vec[i];
                                                //         }
                                                //         right_size = right_size_sum/right_size_vec.size();
                                                // }
                                                
                                                // if(left_size < 65 && right_size < 65 && left_interval != -1 && right_interval != -1){
                                                //         if(left_size > right_size){
                                                //                 y_goal_ = -0.23; 
                                                //         }    
                                                //         else{
                                                //                 y_goal_ = 0.23;
                                                //         }
                                                // }
                                                //x 0으로 주는거 해보기
                                                if(left_size > right_size){
                                                       // if(msg_count_left >= 7){// && my_dir == -1){
                                                                if((left_interval > left_min_interval && left_interval < left_max_interval) || left_interval == -1){//go straight condition
                                                                        y_goal_ += 0;        
                                                                }
                                                                else{        //** left lane tracking condition
                                                                     //   x_goal_ = 0;
                                                                        if(left_interval <= left_min_interval){//need right rotation(ang_vel<0 : right rotation)
                                                                                if(abs(left_interval - left_min_interval) < 10) y_goal_ = -0.05;
                                                                                else if(abs(left_interval - left_min_interval) >= 10 && abs(left_interval - left_min_interval) < 20) y_goal_ = -0.1;
                                                                                else y_goal_ = -0.15;        
                                                                        }
                                                                        else{//need left rotation(ang_vel>0 : left rotation)
                                                                                if(abs(left_interval - left_max_interval) < 10) y_goal_ = 0.05;
                                                                                else if(abs(left_interval - left_max_interval) >= 10 && abs(left_interval - left_max_interval) < 20) y_goal_= 0.1;
                                                                                else y_goal_ = 0.15;
                                                                        }
                                                                }
                                                      //  }
                                                }
                                                else{
                                                       // if(msg_count_right >= 7){// && my_dir == 1){
                                                                if((right_interval > right_min_interval && right_interval < right_max_interval) || right_interval == -1){//go straight condition
                                                                        y_goal_ += (float)0.0;        
                                                                }
                                                                else{
                                                                        //** right lane tracking condition
                                                                      //  x_goal_ = 0;
                                                                        if(right_interval <= right_min_interval){//need left rotation(ang_vel>0 : left rotation)
                                                                                if(abs(right_interval - right_min_interval) < 10) y_goal_ += (float)0.05;
                                                                                else if(abs(left_interval - left_min_interval) >= 10 && abs(left_interval - left_min_interval) < 20) y_goal_ += (float)0.1;
                                                                                else y_goal_ += (float)0.15;
                                                                        }
                                                                        else{//need right rotation(ang_vel<0 : right rotation)
                                                                                if(abs(right_interval - right_max_interval) < 10) y_goal_ += (float)-0.05;
                                                                                else if(abs(left_interval - left_max_interval) >= 10 && abs(left_interval - left_max_interval) < 20) y_goal_ += (float)-0.1;
                                                                                else y_goal_ += (float)-0.15;
                                                                        }
                                                                }     
                                                       // }
                                                }
                                                
                                                
                                                if(left_interval == -1 && right_interval == -1){
                                                        y_goal_ = prev_y_goal_;
                                                }
                                                

                                        }
                                        else if(!right_lane_fitting.empty() && left_lane_fitting.empty()){//tracking right lane(white lane)
                                                
                                                //if(msg_count_right >= 7){
                                                        // int right_size = -1;
                                                        // int right_size_sum = 0;
                                                        // if(!right_size_vec.empty()){
                                                                
                                                        //         for(int i = 0; i<right_size_vec.size(); i++){
                                                        //                 right_size_sum += right_size_vec[i];
                                                        //         }
                                                        //         right_size = right_size_sum/right_size_vec.size();
                                                                
                                                        // }
                                                        // if(pre_right_interval != -1 && right_size > 70){
                                                        //        // if(abs(pre_right_interval - right_interval) > 15) right_interval = -1;//right_min_interval +1;
                                                        // }
                                                        
                                                        if((right_interval > right_min_interval && right_interval < right_max_interval) || right_interval == -1){//go straight condition
                                                                y_goal_ = 0;        
                                                        }
                                                        else{
                                                                //** right lane tracking condition
                                                               // x_goal_ = 0;
                                                                if(right_interval <= right_min_interval){//need left rotation(ang_vel>0 : left rotation)
                                                                        if(abs(right_interval - right_min_interval) <= 10) y_goal_ = 0.13;
                                                                        else if(abs(right_interval - right_min_interval) > 10 && abs(right_interval - right_min_interval) <= 20) y_goal_ = 0.16;
                                                                        else y_goal_ = 0.26;
                                                                }
                                                                else{//need right rotation(ang_vel<0 : right rotation)
                                                                        if(abs(right_interval - right_max_interval) <= 10) y_goal_ = -0.13;
                                                                        else if(abs(right_interval - right_max_interval) > 10 && abs(right_interval - right_max_interval) <= 20) y_goal_ = -0.2;
                                                                        else y_goal_ = -0.26;
                                                                
                                                                }
                                                        }
                                                        // int right_size = -1;
                                                        //     int right_size_sum = 0;
                                                        // if(!right_size_vec.empty()){
                                                            
                                                        //     for(int i = 0; i<right_size_vec.size(); i++){
                                                        //             right_size_sum += right_size_vec[i];
                                                        //     }
                                                        //     right_size = right_size_sum/right_size_vec.size();
                                                        //         // if( right_size > 65 && right_size <= 80  && right_interval > right_min_interval && right_interval < right_max_interval){
                                                        //         //     //    y_goal_ = 0.1; 
                                                        //         // }
                                                        //         // else if(right_size <= 65 || (right_interval == -1 && left_interval == -1)){
                                                        //         //        y_goal_ = 0.13;
                                                        //         // }
                                                        // }
                                                        
                                                        int right_size = right_lane_fitting.size();
                                                        if(right_interval < 40){//} || right_size <40 ){//|| right_slope < 0.38){
                                                                y_goal_ = 0.23;
                                                               // curve_mode = true;
                                                              //  curve_stage = 0;
                                                        }    
                                                        if(pre_right_interval == -1){
                                                                if(abs(pre_right_interval-right_interval)>90) y_goal_ = prev_y_goal_;
                                                        } 
                                                         if(left_interval == -1 && right_interval == -1){
                                                                y_goal_ = prev_y_goal_;
                                                        }
                                                }
                                                // if(right_slope > -0.44){
                                                //         x_goal_ = 0;
                                                //         y_goal_ = 0.5;
                                                // }
                                                // else{
                                                //         y_goal_ = 0;
                                                // }


                                                // if(right_lane_fitting.size()>65 && right_lane_fitting.size() <= 80 && right_interval > right_min_interval && right_interval < right_max_interval){
                                                //         y_goal_ = 0.28;  
                                                // }
                                                // else if(right_lane_fitting.size() <= 65){
                                                //         y_goal_ = 0.33;
                                                // }
                                                //int right_dir = 0;
                                                //right_dir = slopeCheck(slope_test,right_lane_fitting,"right");
                                                // if(right_direction_cnt >3 && right_interval > right_min_interval && right_interval < right_max_interval){
                                                //         y_goal_ = 0.28;
                                                //         right_direction_cnt = 0;
                                                // }
                                                
                                                
                                      //  }
                                        else if(!left_lane_fitting.empty() && right_lane_fitting.empty()){
                                                
                                               // if(msg_count_left >= 7){
                                                        // int left_size = -1;
                                                        // int left_size_sum = 0;
                                                        // if(!left_size_vec.empty()){
                                                                
                                                        //         for(int i = 0; i<left_size_vec.size(); i++){
                                                        //                 left_size_sum += left_size_vec[i];
                                                        //         }
                                                        //         left_size = left_size_sum/left_size_vec.size();
                                                                
                                                        // }
                                                        if(pre_left_interval != -1 && left_lane_fitting.size() > 70){
                                                              //  if(abs(pre_left_interval - left_interval) > 15) left_interval = -1;//left_min_interval + 1;
                                                        }
                                                        // if(pre_left_interval != -1 && left_interval == -1){
                                                        //         left_interval = pre_left_interval;
                                                        // }
                                                       // if(left_interval == -1) left_interval = left_min_interval + 1;
                                                        if((left_interval > left_min_interval && left_interval < left_max_interval) || left_interval == -1){//go straight condition
                                                                y_goal_ = 0;        
                                                        }
                                                        else{
                                                                //** left lane tracking condition
                                                                //x_goal_ = 0;
                                                                if(left_interval <= left_min_interval){//need right rotation(ang_vel<0 : right rotation)
                                                                        if(abs(left_interval - left_min_interval) <= 10) y_goal_ = -0.1;
                                                                        else if(abs(left_interval - left_min_interval) >= 10 && abs(left_interval - left_min_interval) < 20) y_goal_ = -0.13;
                                                                        else y_goal_ = -0.16;        
                                                                }
                                                                else{//need left rotation(ang_vel>0 : left rotation)
                                                                        if(abs(left_interval - left_max_interval) <= 10) y_goal_ = 0.1;
                                                                        else if(abs(left_interval - left_max_interval) >= 10 && abs(left_interval - left_max_interval) < 20) y_goal_ = 0.13;
                                                                        else y_goal_ = 0.16;
                                                                        
                                                                }
                                                        }
                                                        // if(!left_size_vec.empty()){
                                                                
                                                        //         if(left_size > 65 && left_size <= 80  && left_interval > left_min_interval && left_interval < left_max_interval){
                                                        //               //  y_goal_ = -0.1; 
                                                        //         }
                                                        //         else if(left_size <= 65){
                                                        //          //       x_goal_ = 0;
                                                        //               //  y_goal_ = -0.13;
                                                        //         }
                                                        // }
                                                        int left_size = left_lane_fitting.size();
                                                        if(left_interval < 40){//|| left_size < 40){
                                                                y_goal_ = -0.23;
                                                                //curve_mode = true;
                                                                //curve_stage = 5;
                                                        }
                                                        if(pre_left_interval == -1){
                                                                if(abs(pre_left_interval-left_interval)>90) y_goal_ = prev_y_goal_;
                                                        } 
                                                         if(left_interval == -1 && right_interval == -1){
                                                                y_goal_ = prev_y_goal_;
                                                        }
                                               // }
                                                
                                                // if(left_lane_fitting.size() > 65 && left_lane_fitting.size() <= 80 && left_interval > left_min_interval && left_interval < left_max_interval){
                                                //         y_goal_ = -0.28;  
                                                // }
                                                // else if(left_lane_fitting.size() <= 65){
                                                //         y_goal_ = -0.33;
                                                // }
                                                // //int left_dir = 0;
                                                // //left_dir = slopeCheck(slope_test,left_lane_fitting,"left");
                                                // if(left_direction_cnt > 3 && left_interval > left_min_interval && left_interval < left_max_interval){
                                                //         y_goal_ = -0.28;
                                                //         left_direction_cnt = 0;
                                                // } 
                                                
                                        }
                                        else{//if detected no lane, than go straight
                                                if(prev_y_goal_ < 0){
                                                        x_goal_ = 0.03;//.08;
                                                        y_goal_ = 0;//13;
                                                }
                                                else if(prev_y_goal_ > 0){
                                                        x_goal_ = 0.03;//.08;
                                                        y_goal_ = 0;//-0.13;
                                                }
                                                else{
                                                        //x_goal_ = 0;
                                                        y_goal_ = 0;
                                                }
                                                
                                                
                                        }
                                        
                                        std::cout<<"x_goal_ : "<<x_goal_<<std::endl;
                                        std::cout<<"y_goal_ : "<<y_goal_<<std::endl;
                                        //std::cout<<"pre_left_interval : "<<pre_left_interval<<std::endl;
                                        std::cout<<"right_interval : "<<right_interval<<std::endl;
                                        std::cout<<"left_interval : "<<left_interval<<std::endl;
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        
                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        prev_y_goal_ = y_goal_;
                                        msg_count_left = 0;
                                        msg_count_right = 0;
                                        left_interval_vec.clear();
                                        right_interval_vec.clear();
                                        left_interval_vec.resize(0);
                                        right_interval_vec.resize(0);
                                        pre_left_interval = left_interval;
                                        pre_right_interval = right_interval;
                                        left_interval = -1;
                                        right_interval = -1;
                                        
                               // }
                        }
                        if(curve_mode){
                                                
                                                switch (curve_stage)
                                                {
                                                        case 0://left_turn by right line
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0;
                                                                y_goal_ = 0;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                curve_stage = 1;/* code */
                                                                go_cnt = 0;
                                                                break;
                                                        case 1:
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0;
                                                                y_goal_ = 0;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                go_cnt++;
                                                                if(go_cnt > 10){
                                                                        curve_stage = 2;
                                                                        go_cnt = 0;
                                                                }
                                                                break;
                                                        case 2:
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0;
                                                                y_goal_ = 0.35;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                go_cnt++;
                                                                if(go_cnt > 15){
                                                                        curve_stage = 3;
                                                                        go_cnt = 0;
                                                                }
                                                                break;
                                                        case 3:
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0.08;
                                                                y_goal_ = 0;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                go_cnt++;
                                                                if(go_cnt > 5){
                                                                        curve_stage = 4;
                                                                        go_cnt = 0;
                                                                }
                                                                break;
                                                        
                                                        case 4:
                                                                curve_mode = false;
                                                                go_cnt = 0;
                                                                break;
                                                        case 5://left_turn by right line
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0;
                                                                y_goal_ = 0;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                curve_stage = 6;/* code */
                                                                go_cnt = 0;
                                                                break;
                                                        case 6:
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0;
                                                                y_goal_ = 0;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                go_cnt++;
                                                                if(go_cnt > 10){
                                                                        curve_stage = 7;
                                                                        go_cnt = 0;
                                                                }
                                                                break;
                                                        case 7:
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0.1;
                                                                y_goal_ = -0.35;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                go_cnt++;
                                                                if(go_cnt > 15){
                                                                        curve_stage = 8;
                                                                        go_cnt = 0;
                                                                }
                                                                break;
                                                        case 8:
                                                                goal_array.data.clear();
                                                                goal_array.data.resize(0);
                                                                x_goal_ = 0.08;
                                                                y_goal_ = 0;
                                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                                goal_array.data.push_back(y_goal_);
                                                                go_cnt++;
                                                                if(go_cnt > 5){
                                                                        curve_stage = 9;
                                                                        go_cnt = 0;
                                                                }
                                                                break;
                                                        
                                                        case 9:
                                                                curve_mode = false;
                                                                go_cnt = 0;
                                                                break;
                                                }

                                        }
                        cv::Mat ss = frame.clone();
                        if(parking_mode){
                                std::vector<cv::Point> object_check;
                                cv::Mat parked = origin_white_hsv.clone();
                                cv::Mat colored_park = frame.clone();
                                switch (parking_stage)
                                {
                                        case 0: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                //hsv
                                                // if(!dot_lane_fitting.empty()){
                                                //         // for(int i = 0; i<dot_lane_fitting.size()-1; i++){
                                                //         //         if(abs(dot_lane_fitting[i].y - dot_lane_fitting[i+1].y) > 6){
                                                //         //                 cv::putText(ss, std::to_string(dot_lane_fitting[i].y), dot_lane_fitting[i], 
                                                //         //                 cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(235,25,255), 2);
                                                //         //                 cv::putText(ss, std::to_string(dot_lane_fitting[i+1].y), dot_lane_fitting[i+1], 
                                                //         //                 cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(235,25,255), 2);
                                                //         //                 cv::line(ss,dot_lane_fitting[i], dot_lane_fitting[i+1],cv::Scalar(200,200,20),2.4);
                                                //         //                 first_point = dot_lane_fitting[i];
                                                //         //         }
                                                //         // }
                                                // }
                                                parking_stage = 1;
                                                break;
                                        case 1: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.05;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 10){
                                                        parking_stage = 2;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 2: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 10){
                                                        parking_stage = 3;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 3: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = -0.25;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 25){
                                                        parking_stage = 4;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 4: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 5;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 5: 
                                                {
                                               
                                                int y_pt = -1, st_x_pt = -1, no_pt = 0;
                                                for(int x = 0; x<parked.cols; x++){
                                                        uchar* park_pt = parked.ptr<uchar>(x);
                                                        no_pt = 0;
                                                        for(int y = parked.rows-1; y>0; y--){
                                                                if(park_pt[y] != (uchar)0){
                                                                        if(no_pt>100){
                                                                                st_x_pt = x;
                                                                                y_pt = y;
                                                                                break;
                                                                        }      
                                                                }
                                                                else{
                                                                        no_pt++;
                                                                }
                                                        }
                                                        if(st_x_pt != -1){
                                                                break;
                                                        }
                                                }
                                                no_pt = 0;
                                                // for(int x = 0; x<parked.cols; x++){
                                                //         uchar* park_pt = parked.ptr<uchar>(x);
                                                //         int no_pt = 0;
                                                //         for(int y = parked.rows-1; y>0; y--){
                                                //                 if(park_pt[y] != (uchar)0){
                                                //                         if(no_pt<5){
                                                //                                 no_pt = 0;
                                                //                                 center_sum++;
                                                //                         }
                                                //                         else{
                                                //                                 break;
                                                //                         }
                                                //                 }
                                                //                 else{
                                                //                         no_pt++;
                                                //                 }
                                                //         }
                                                //         if(abs(pre_center_sum - center_sum) > 60 && pre_center_sum != -1){
                                                //                 park_center_x = x;
                                                //                 break;
                                                //         }
                                                //         pre_center_sum = center_sum;
                                                //         center_sum = 0;
                                                // }
                                                //if(center_sum != -1){
                                                        // for(int y = y_pt; x<parked.cols; x++){
                                                int good_pt = 0;

                                                uchar* park_pt = parked.ptr<uchar>(y_pt - 2);
                                                for(int x = st_x_pt; x<parked.cols; x++){
                                                        if(park_pt[x] == (uchar)0){
                                                                no_pt++;
                                                                if(no_pt > 10){
                                                                        break;
                                                                }
                                                                else{
                                                                        no_pt = 0;
                                                                }
                                                        }
                                                        else{
                                                                good_pt++;
                                                        }
                                                        //std::cout<<"check x : "<<x<<std::endl;
                                                }
                                                
                                                if(no_pt <= 10 || good_pt > parked.cols/2){
                                                        parking_stage = 6;
                                                }
                                                else{
                                                        parking_stage = 13;
                                                }
                                                        // }
                                                        // for(int x = park_center_x; x<parked.cols; x++){
                                                        //         uchar* park_pt = parked.ptr<uchar>(x);
                                                        //         for(int y = parked.rows-1; y>0; y--){
                                                        //                 if(park_pt[y] != (uchar)0){
                                                        //                         object_check.push_back(cv::Point(x,y));
                                                        //                         park_pt[y*parked.channels()+1] = 50;
                                                        //                         std::cout<<"object_check(x,y) : "<<cv::Point(x,y)<<std::endl;
                                                        //                 }
                                                        //         }
                                                        // }
                                                        // int check_y = 0;
                                                        // for(int i = 0; i<object_check.size()-1; i++){
                                                        //         if(abs(object_check[i].y - object_check[i+1].y) > 50){
                                                        //                 if(abs(object_check[i].x - object_check[i+1].x)<3){
                                                        //                         check_y++;
                                                        //                 }
                                                                        
                                                        //         }
                                                        // }
                                                        
                                                        // if(check_y > 6){
                                                        //         parking_stage = 6;
                                                                
                                                        // }
                                                        // else{
                                                        //         parking_stage = 13;
                                                        // }
                                                //}
                                                }
                                                break;
                                        case 6://first parking area
                                                {
                                                if(go_cnt<20){
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0.05;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        go_cnt++;
                                                }
                                                else{
                                                        go_cnt = 0;
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        
                                                        parking_stage = 7;
                                                        
                                                }
                                                }
                                                break;
                                        case 7 ://first parking area
                                                {
                                                if(go_cnt<5){
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        go_cnt++;
                                                }
                                                else{
                                                        go_cnt = 0;
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        
                                                        parking_stage = 8;
                                                        
                                                }
                                                }
                                                break;        
                                        case 8 :
                                                if(go_cnt<20){
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = -0.05;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        go_cnt++;
                                                }
                                                else{
                                                        go_cnt = 0;
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        parking_stage = 9;
                                                       
                                                }
                                                break;
                                        case 9 : 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 10){

                                                        parking_stage = 10;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 10 : 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0.25;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 25){

                                                        parking_stage = 11;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 11 : 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 10){
                                                        go_cnt = 0;
                                                        parking_stage = 12;
                                                }
                                                
                                                break;
                                        case 12: 
                                                normal_mode = true;
                                                parking_mode = false;
                                                parking_checked = true;
                                                break;///////////
                                        case 13:
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 14;
                                                        go_cnt = 0;
                                                }
                                                break; 
                                        case 14: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0.25;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 25){

                                                        parking_stage = 15;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 15: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 16;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 16: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.05;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 20){
                                                        parking_stage = 17;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 17: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 18;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 18: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = -0.25;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 25){
                                                        parking_stage = 19;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 19: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 20;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 20: 
                                                {
                                                if(go_cnt<20){
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0.05;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        go_cnt++;
                                                }
                                                else{
                                                        go_cnt = 0;
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        parking_stage = 21;
                                                        
                                                }
                                                }
                                                break;
                                        case 21: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 22;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 22:
                                                if(go_cnt<20){
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = -0.05;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        go_cnt++;
                                                }
                                                else{
                                                        go_cnt = 0;
                                                        goal_array.data.clear();
                                                        goal_array.data.resize(0);
                                                        x_goal_ = 0;
                                                        y_goal_ = 0;
                                                        goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                        goal_array.data.push_back(y_goal_);
                                                        parking_stage = 23;
                                                       
                                                }
                                                break;
                                        case 23: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 10){

                                                        parking_stage = 24;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 24: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0.25;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 25){

                                                        parking_stage = 25;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 25: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        parking_stage = 26;
                                                }
                                                
                                                break;
                                        case 26: 
                                                normal_mode = true;
                                                parking_mode = false;
                                                parking_checked = true;
                                                go_cnt = 0;
                                                break;

                                                
                                
                                }
                                cv::imshow("park",parked);
                                
                                //parking_checked = true;
                               
                        }
                        //cv::imshow("ss",ss);
                        if(blocking_bar_mode){
                                switch(blocking_bar_stage){
                                        case 0: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.05;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 10){
                                                        blocking_bar_stage = 1;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 1: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 30 && detectBlockingBar(red_hsv)){
                                                        blocking_bar_stage = 3;
                                                        go_cnt = 0;
                                                }
                                                else if(go_cnt > 30 && !detectBlockingBar(red_hsv)){
                                                        blocking_bar_stage = 5;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 2: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.05;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        blocking_bar_stage = 3;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 3:
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        blocking_bar_stage = 4;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 4:
                                                normal_mode = true;
                                                blocking_bar_checked = true;
                                                blocking_bar_mode = false;
                                                go_cnt = 0;
                                                break;
                                        case 5:
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.05;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 5){
                                                        blocking_bar_stage = 6;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        case 6: 
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);//linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                                if(go_cnt > 30 && !detectBlockingBar(red_hsv)){
                                                        blocking_bar_stage = 2;
                                                        go_cnt = 0;
                                                }
                                                break;
                                        
                                }
                        }
                        
                      
                        
                        
                        ////*Restore birdeyeview img to origin view*////
                        restoreImgWithLangeMerge(callane,frame,yellow_labeling,white_labeling,mergelane);
                        
                        ////*Make lane infomation msg for translate scan data*////
                        extractLanePoint(origin,mergelane);

                        output_origin_for_copy = origin.clone();        
                        // left_point = false;
                        // right_point = false;
                        
                        // left_theta = 0;
                        // right_theta = 0;
                        // if(is_left_box_true && is_right_box_true){
                        //         center_cali = abs(left_lane_fitting[20].x - right_lane_fitting[20].x)/2;
                        // }
                        // else if(is_left_box_true){
                        //         center_cali = left_lane_fitting[20].x + 50;
                        // }
                        // else{
                        //         center_cali = right_lane_fitting[20].x - 50;
                        // }
                        
                }
                else{//frame is empty
                        while(frame.empty()) {//for unplugged camera
                                cv_ptr = cv_bridge::toCvCopy(img_msg,sensor_msgs::image_encodings::BGR8);
                                frame = cv_ptr->image;
                        }
                        x_goal_ = 0;
                        y_goal_ = 0.;//제자리회전시켜보기
                        center_cali = -1;
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

        nh.param<int>("/"+groupName+"/lane_detection/reset_msg",reset_msg,0);
        //default yellow when no launch file for lane color
        nh.param<int>("/"+groupName+"/lane_detection/y_hmin",y_hmin,15);
        nh.param<int>("/"+groupName+"/lane_detection/y_hmax",y_hmax,21);
        nh.param<int>("/"+groupName+"/lane_detection/y_smin",y_smin,52);
        nh.param<int>("/"+groupName+"/lane_detection/y_smax",y_smax,151);
        nh.param<int>("/"+groupName+"/lane_detection/y_vmin",y_vmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/y_vmax",y_vmax,180);
        //default white when no launch file for lane color
        nh.param<int>("/"+groupName+"/lane_detection/w_hmin",w_hmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/w_hmax",w_hmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/w_smin",w_smin,0);
        nh.param<int>("/"+groupName+"/lane_detection/w_smax",w_smax,24);
        nh.param<int>("/"+groupName+"/lane_detection/w_vmin",w_vmin,172);
        nh.param<int>("/"+groupName+"/lane_detection/w_vmax",w_vmax,255);

        //default red when no launch file for signal lamp
        nh.param<int>("/"+groupName+"/lane_detection/r_hmin",r_hmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/r_hmax",r_hmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/r_smin",r_smin,0);
        nh.param<int>("/"+groupName+"/lane_detection/r_smax",r_smax,24);
        nh.param<int>("/"+groupName+"/lane_detection/r_vmin",r_vmin,172);
        nh.param<int>("/"+groupName+"/lane_detection/r_vmax",r_vmax,255);

        //default red when no launch file for signal lamp
        nh.param<int>("/"+groupName+"/lane_detection/r2_hmin",r2_hmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/r2_hmax",r2_hmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/r2_smin",r2_smin,0);
        nh.param<int>("/"+groupName+"/lane_detection/r2_smax",r2_smax,24);
        nh.param<int>("/"+groupName+"/lane_detection/r2_vmin",r2_vmin,172);
        nh.param<int>("/"+groupName+"/lane_detection/r2_vmax",r2_vmax,255);
        //default red when no launch file for signal lamp
        nh.param<int>("/"+groupName+"/lane_detection/y2_hmin",y2_hmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/y2_hmax",y2_hmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/y2_smin",y2_smin,0);
        nh.param<int>("/"+groupName+"/lane_detection/y2_smax",y2_smax,24);
        nh.param<int>("/"+groupName+"/lane_detection/y2_vmin",y2_vmin,172);
        nh.param<int>("/"+groupName+"/lane_detection/y2_vmax",y2_vmax,255);
        //default red when no launch file for signal lamp
        nh.param<int>("/"+groupName+"/lane_detection/g_hmin",g_hmin,0);
        nh.param<int>("/"+groupName+"/lane_detection/g_hmax",g_hmax,180);
        nh.param<int>("/"+groupName+"/lane_detection/g_smin",g_smin,0);
        nh.param<int>("/"+groupName+"/lane_detection/g_smax",g_smax,24);
        nh.param<int>("/"+groupName+"/lane_detection/g_vmin",g_vmin,172);
        nh.param<int>("/"+groupName+"/lane_detection/g_vmax",g_vmax,255);
        ROS_INFO("lane_detection %d %d %d %d %d %d %d", debug, web_cam, imshow, track_bar, time_check, lable, gazebo);
        ROS_INFO("imshow %d", imshow);
}

void InitImgObjectforROS::initMyRESETTrackbar(const string &trackbar_name){
        cv::namedWindow(trackbar_name,cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("reset msg",trackbar_name, &reset_msg, 1, NULL);
        cv::setTrackbarPos("reset msg",trackbar_name, reset_msg);
}
void InitImgObjectforROS::setMyRESETTrackbarValue(const string &trackbar_name){
        reset_msg = cv::getTrackbarPos("reset msg",trackbar_name);
        nh.setParam("/"+groupName+"/lane_detection/reset_msg",reset_msg);
}

void InitImgObjectforROS::initMyHSVTrackbar(const string &trackbar_name){
                
                cv::namedWindow(trackbar_name, cv::WINDOW_AUTOSIZE);
                if (trackbar_name.find("YELLOW") != string::npos && trackbar_name.find("LANE") != string::npos) { 

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
                else if(trackbar_name.find("WHITE") != string::npos && trackbar_name.find("LANE") != string::npos){

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
                else if(trackbar_name.find("BLOCKING") != string::npos && trackbar_name.find("RED") != string::npos){

                        cv::createTrackbar("h min", trackbar_name, &r_hmin, 179, NULL);
                        cv::setTrackbarPos("h min", trackbar_name, r_hmin);

                        cv::createTrackbar("h max", trackbar_name, &r_hmax, 179, NULL);
                        cv::setTrackbarPos("h max", trackbar_name, r_hmax);

                        cv::createTrackbar("s min", trackbar_name, &r_smin, 255, NULL);
                        cv::setTrackbarPos("s min", trackbar_name, r_smin);

                        cv::createTrackbar("s max", trackbar_name, &r_smax, 255, NULL);
                        cv::setTrackbarPos("s max", trackbar_name, r_smax);

                        cv::createTrackbar("v min", trackbar_name, &r_vmin, 255, NULL);
                        cv::setTrackbarPos("v min", trackbar_name, r_vmin);

                        cv::createTrackbar("v max", trackbar_name, &r_vmax, 255, NULL);
                        cv::setTrackbarPos("v max", trackbar_name, r_vmax);
                }
                else if(trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("RED") != string::npos ){

                        cv::createTrackbar("h min", trackbar_name, &r2_hmin, 179, NULL);
                        cv::setTrackbarPos("h min", trackbar_name, r2_hmin);

                        cv::createTrackbar("h max", trackbar_name, &r2_hmax, 179, NULL);
                        cv::setTrackbarPos("h max", trackbar_name, r2_hmax);

                        cv::createTrackbar("s min", trackbar_name, &r2_smin, 255, NULL);
                        cv::setTrackbarPos("s min", trackbar_name, r2_smin);

                        cv::createTrackbar("s max", trackbar_name, &r2_smax, 255, NULL);
                        cv::setTrackbarPos("s max", trackbar_name, r2_smax);

                        cv::createTrackbar("v min", trackbar_name, &r2_vmin, 255, NULL);
                        cv::setTrackbarPos("v min", trackbar_name, r2_vmin);

                        cv::createTrackbar("v max", trackbar_name, &r2_vmax, 255, NULL);
                        cv::setTrackbarPos("v max", trackbar_name, r2_vmax);
                }
                else if(trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("YELLOW") != string::npos){

                        cv::createTrackbar("h min", trackbar_name, &y2_hmin, 179, NULL);
                        cv::setTrackbarPos("h min", trackbar_name, y2_hmin);

                        cv::createTrackbar("h max", trackbar_name, &y2_hmax, 179, NULL);
                        cv::setTrackbarPos("h max", trackbar_name, y2_hmax);

                        cv::createTrackbar("s min", trackbar_name, &y2_smin, 255, NULL);
                        cv::setTrackbarPos("s min", trackbar_name, y2_smin);

                        cv::createTrackbar("s max", trackbar_name, &y2_smax, 255, NULL);
                        cv::setTrackbarPos("s max", trackbar_name, y2_smax);

                        cv::createTrackbar("v min", trackbar_name, &y2_vmin, 255, NULL);
                        cv::setTrackbarPos("v min", trackbar_name, y2_vmin);

                        cv::createTrackbar("v max", trackbar_name, &y2_vmax, 255, NULL);
                        cv::setTrackbarPos("v max", trackbar_name, y2_vmax);
                }
                else if(trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("GREEN") != string::npos){

                        cv::createTrackbar("h min", trackbar_name, &g_hmin, 179, NULL);
                        cv::setTrackbarPos("h min", trackbar_name, g_hmin);

                        cv::createTrackbar("h max", trackbar_name, &g_hmax, 179, NULL);
                        cv::setTrackbarPos("h max", trackbar_name, g_hmax);

                        cv::createTrackbar("s min", trackbar_name, &g_smin, 255, NULL);
                        cv::setTrackbarPos("s min", trackbar_name, g_smin);

                        cv::createTrackbar("s max", trackbar_name, &g_smax, 255, NULL);
                        cv::setTrackbarPos("s max", trackbar_name, g_smax);

                        cv::createTrackbar("v min", trackbar_name, &g_vmin, 255, NULL);
                        cv::setTrackbarPos("v min", trackbar_name, g_vmin);

                        cv::createTrackbar("v max", trackbar_name, &g_vmax, 255, NULL);
                        cv::setTrackbarPos("v max", trackbar_name, g_vmax);
                }
}

void InitImgObjectforROS::setMyHSVTrackbarValue(const string &trackbar_name){

                if (trackbar_name.find("YELLOW") != string::npos && trackbar_name.find("LANE") != string::npos) { 
                        y_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        y_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        y_smin = cv::getTrackbarPos("s min", trackbar_name);
                        y_smax = cv::getTrackbarPos("s max", trackbar_name);
                        y_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        y_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }
                else if(trackbar_name.find("WHITE") != string::npos && trackbar_name.find("LANE") != string::npos){
                        w_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        w_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        w_smin = cv::getTrackbarPos("s min", trackbar_name);
                        w_smax = cv::getTrackbarPos("s max", trackbar_name);
                        w_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        w_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }
                else if(trackbar_name.find("BLOCKING_RED") != string::npos){

                        r_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        r_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        r_smin = cv::getTrackbarPos("s min", trackbar_name);
                        r_smax = cv::getTrackbarPos("s max", trackbar_name);
                        r_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        r_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }
                else if(trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("RED") != string::npos){

                        r2_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        r2_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        r2_smin = cv::getTrackbarPos("s min", trackbar_name);
                        r2_smax = cv::getTrackbarPos("s max", trackbar_name);
                        r2_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        r2_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }
                else if(trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("RED") != string::npos){

                        y2_hmin = cv::getTrackbarPos("h min", trackbar_name); 
                        y2_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        y2_smin = cv::getTrackbarPos("s min", trackbar_name);
                        y2_smax = cv::getTrackbarPos("s max", trackbar_name);
                        y2_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        y2_vmax = cv::getTrackbarPos("v max", trackbar_name);
                }
                else if(trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("GREEN") != string::npos){

                        g_hmin = cv::getTrackbarPos("h min", trackbar_name);
                        g_hmax = cv::getTrackbarPos("h max", trackbar_name);
                        g_smin = cv::getTrackbarPos("s min", trackbar_name);
                        g_smax = cv::getTrackbarPos("s max", trackbar_name);
                        g_vmin = cv::getTrackbarPos("v min", trackbar_name);
                        g_vmax = cv::getTrackbarPos("v max", trackbar_name);
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


                nh.setParam("/"+groupName+"/lane_detection/r_hmin",r_hmin);
                nh.setParam("/"+groupName+"/lane_detection/r_hmax",r_hmax);
                nh.setParam("/"+groupName+"/lane_detection/r_smin",r_smin);
                nh.setParam("/"+groupName+"/lane_detection/r_smax",r_smax);
                nh.setParam("/"+groupName+"/lane_detection/r_vmin",r_vmin);
                nh.setParam("/"+groupName+"/lane_detection/r_vmax",r_vmax);


                nh.setParam("/"+groupName+"/lane_detection/r2_hmin",r2_hmin);
                nh.setParam("/"+groupName+"/lane_detection/r2_hmax",r2_hmax);
                nh.setParam("/"+groupName+"/lane_detection/r2_smin",r2_smin);
                nh.setParam("/"+groupName+"/lane_detection/r2_smax",r2_smax);
                nh.setParam("/"+groupName+"/lane_detection/r2_vmin",r2_vmin);
                nh.setParam("/"+groupName+"/lane_detection/r2_vmax",r2_vmax);

                nh.setParam("/"+groupName+"/lane_detection/g_hmin",g_hmin);
                nh.setParam("/"+groupName+"/lane_detection/g_hmax",g_hmax);
                nh.setParam("/"+groupName+"/lane_detection/g_smin",g_smin);
                nh.setParam("/"+groupName+"/lane_detection/g_smax",g_smax);
                nh.setParam("/"+groupName+"/lane_detection/g_vmin",g_vmin);
                nh.setParam("/"+groupName+"/lane_detection/g_vmax",g_vmax);

                nh.setParam("/"+groupName+"/lane_detection/y2_hmin",y2_hmin);
                nh.setParam("/"+groupName+"/lane_detection/y2_hmax",y2_hmax);
                nh.setParam("/"+groupName+"/lane_detection/y2_smin",y2_smin);
                nh.setParam("/"+groupName+"/lane_detection/y2_smax",y2_smax);
                nh.setParam("/"+groupName+"/lane_detection/y2_vmin",y2_vmin);
                nh.setParam("/"+groupName+"/lane_detection/y2_vmax",y2_vmax);

             
}

void InitImgObjectforROS::setColorPreocessing(lane_detect_algo::CalLane callane, cv::Mat src, cv::Mat& dst_y, cv::Mat& dst_w, cv::Mat& dst_r, cv::Mat& dst_r2, cv::Mat& dst_g, cv::Mat& dst_y2){
                ////*Make trackbar obj*////
                if(track_bar){
                        setMyHSVTrackbarValue(groupName+"_YELLOW_LANE_TRACKBAR");
                        setMyHSVTrackbarValue(groupName+"_WHITE_LANE_TRACKBAR");
                        setMyHSVTrackbarValue(groupName+"_BLOCKING_RED_TRACKBAR");
                        setMyHSVTrackbarValue(groupName+"_SIGNAL_RED_TRACKBAR");
                        setMyHSVTrackbarValue(groupName+"_SIGNAL_GREEN_TRACKBAR");
                        setMyHSVTrackbarValue(groupName+"_SIGNAL_YELLOW_TRACKBAR");;
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
                        if(debug) cv::imshow("bev_le",bev);
                }
                else if(groupName == "right"){
                        
                        callane.birdEyeView_right(src,bev);
                        if(debug) cv::imshow("bev_ri",bev);
                }
                

                ////*Detect yellow and white colors and make dst img to binary img by hsv value*////
                if (track_bar) {//Use trackbar. Use real-time trackbar's hsv value
                        //for detect lane color
                        callane.detectYHSVcolor(bev, dst_y, y_hmin, y_hmax, y_smin, y_smax, y_vmin, y_vmax);
                        callane.detectWhiteRange(bev,dst_w, w_hmin, w_hmax, w_smin, w_smax, w_vmin, w_vmax,0,0);
                        //for detect blocking bar color
                        callane.detectWhiteRange(bev,dst_r, r_hmin, r_hmax, r_smin, r_smax, r_vmin, r_vmax,0,0);
                        //for detect signal lamp
                        callane.detectWhiteRange(bev,dst_r2, r2_hmin, r2_hmax, r2_smin, r2_smax, r2_vmin, r2_vmax,0,0);
                        callane.detectWhiteRange(bev,dst_g, g_hmin, g_hmax, g_smin, g_smax, g_vmin, g_vmax,0,0);
                        callane.detectWhiteRange(bev,dst_y2, y2_hmin, y2_hmax, y2_smin, y2_smax, y2_vmin, y2_vmax,0,0);

                        cv::imshow(groupName+"_YELLOW_LANE_TRACKBAR",dst_y);
                        cv::imshow(groupName+"_WHITE_LANE_TRACKBAR",dst_w);

                        cv::imshow(groupName+"_BLOCKING_RED_TRACKBAR",dst_r);

                        cv::imshow(groupName+"_SIGNAL_RED_TRACKBAR",dst_r2);
                         cv::imshow(groupName+"_SIGNAL_GREEN_TRACKBAR",dst_g);
                        cv::imshow(groupName+"_SIGNAL_YELLOW_TRACKBAR",dst_y2);
                       
                        
                }
                else {//Don't use trackbar. Use defalut value.
                        callane.detectYHSVcolor(bev, dst_y, 7, 21, 52, 151, 0, 180);
                        callane.detectWhiteRange(bev, dst_w, 0, 180, 0, 29, 179, 255,0,0);

                        callane.detectWhiteRange(bev, dst_r, 160, 179, 0, 29, 179, 255,0,0);

                        callane.detectWhiteRange(bev, dst_r2, 160, 179, 0, 29, 179, 255,0,0);
                        callane.detectWhiteRange(bev, dst_g, 38, 75, 0, 29, 179, 255,0,0);
                        callane.detectWhiteRange(bev, dst_y2, 7, 21, 52, 151, 0, 180,0,0);
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
                // center_pt_t = cv::Point(center_pt_t.x*2,center_pt_b.y*2);
                // center_pt_b = cv::Point(center_pt_b.x*2,center_pt_b.y*2);
                // center_cali = center_cali*2;
                
                // if(center_cali != -1){
                //         cv::line(output_origin,center_pt_t,center_pt_b,cv::Scalar(100,100,200),2);
                //         cv::line(output_origin,cv::Point(center_cali,center_pt_t.y),cv::Point(center_cali,center_pt_b.y),cv::Scalar(200,100,100),2);
                // }
                cv::line(output_origin, cv::Point(0,423), cv::Point(output_origin.cols-1,423),cv::Scalar(23,32,100),2);
                //cv::line(output_origin, cv::Point(529,310), cv::Point(529,325),cv::Scalar(40,26,200),2);
                //cv::line(output_origin, cv::Point(564,310), cv::Point(564,325),cv::Scalar(40,26,200),2);
                cv::imshow(groupName+"_colorfulLane",output_origin);
}


void InitImgObjectforROS::setPixelGoal(double* goal, int num){
        //  goal_array.data.clear();
        //  goal_array.data.push_back(goal[num]);
        //  goal_array.data.push_back(goal[num+1]);
        //  std::cout<<"==========goal visible :"<<goal_array.data[0]<<", "<<goal_array.data[1]<<std::endl;
        }

void InitImgObjectforROS::setMySobelwithROI(cv::Mat src, double delete_row_per, const string &lane_name, cv::Mat& dst){
        cv::cvtColor(src,dst,CV_BGR2GRAY);
        cv::Mat dst_h = dst.clone();
        cv::Mat dst_v = dst.clone();
        cv::Mat element(1,1,CV_8U,cv::Scalar(1)); 
        cv::Sobel(dst_h,dst_h,dst_h.depth(),0,1);//horizontal
        cv::Sobel(dst_v,dst_v,dst_v.depth(),1,0);//vertical
        cv::Mat sobel_dilate = dst_h | dst_v;
        cv::dilate(sobel_dilate,sobel_dilate,element);
        dst = sobel_dilate;
        cv::threshold(dst, dst, 240, 255, cv::THRESH_BINARY);
        cv::medianBlur(dst, dst, 1);
        for(int y = 0; y<dst.rows/2 ; y++){
                uchar* none_roi_data = dst.ptr<uchar>(y);
                for(int x = 0; x<dst.cols; x++){
                        if(none_roi_data[x] != (uchar)0){
                                none_roi_data[x] = (uchar)0;
                        }
                }
        }
        if(lane_name == "left"){//**set roi for left lane
                for(int y = 0; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = dst.cols-1; x > dst.cols/2; x--){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for(int y = 0; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x < 11; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
        }
        
        if(lane_name == "right"){
                for(int y = 0; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x < dst.cols/2; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
        }
        
                
}
void InitImgObjectforROS::setMyLaneBox(cv::Point t_pt, cv::Point b_pt, const string &lane_name, std::vector<cv::Point> &dst){
        if(!dst.empty() && lane_name == "left"){
                if(dst[1].x > b_pt.x) dst[1].x = b_pt.x;
                if(dst[0].y < t_pt.y) dst[0].y = t_pt.y;
        }
        else if(!dst.empty() && lane_name == "right"){
                if(dst[0].x < t_pt.x) dst[0].x = t_pt.x;
                if(dst[0].y < t_pt.y) dst[0].y = t_pt.y;
        }
}
bool InitImgObjectforROS::setMyLaneFitting(cv::Mat& src_img, std::vector<cv::Point> src_pt, const string &lane_name, std::vector<cv::Point> &dst){
       
        bool return_val = true;
        if(lane_name == "left"){
                for(int y = src_pt[1].y; y > src_pt[0].y; y--) {//
                        uchar* fitting_data = src_img.ptr<uchar>(y);
                        for(int x = src_pt[1].x; x > src_pt[0].x; x--) {                            
                                if(fitting_data[x]!= (uchar)0) {
                                        // if(x == src_pt[1].x){//left는 캐니엣지니까 이거 쓰면 안돼.
                                        //         int my_sum = 0;
                                        //         for(uint i = x; i > x-10; i--){
                                        //                 if(fitting_data[i] != (uchar)0){
                                        //                         my_sum++;
                                        //                 }
                                        //         }
                                        //         if(my_sum > 5){
                                        //                 dst.push_back(cv::Point(x,y));
                                        //                 if(dst.size()>20){
                                        //                        return true; 
                                        //                 } 
                                        //         }
                                        // }
                                        dst.push_back(cv::Point(x,y));
                                        break;
                                }
                        }
                }
                // if(!dst.empty()){
                //         int cnt = 0, diff_sum = 0, diff_avg = 0, noise_sum = 0, noise_avg = 0;
                //         if(dst.size()>10){
                //                 for(uint i = 0; i<9; i++){
                //                         noise_sum += dst[i].x;
                //                 }
                //                 noise_avg = noise_sum/9;
                //                 for(uint i = 0; i<9; i++){
                //                         if(dst[i].x < noise_avg){//이건 처음 영상에 노이즈가 엄청 많을 경우에는 적용될 수 없는 알고리즘..
                //                                 diff_sum += abs(dst[i].x - dst[i+1].x);
                //                                 cnt++;
                //                         }
                //                 }
                //                 diff_avg = diff_sum/cnt;
                //         }
                //         uchar* delete_data;
                //         for(uint i = 0; i<dst.size()-1; i++){
                //                 if(abs(dst[i].x - dst[i+1].x) > diff_avg){
                //                         delete_data = src_img.ptr<uchar>(dst[i+1].y);
                //                         for(int j = dst[i+1].x; j>0; j--){
                //                                 delete_data[j] = (uchar)0;
                //                         }
                //                         dst.erase(dst.begin()+i+1);
                //                 }
                //         }
                        
                // }
        }
        if(lane_name == "right"){
                for(int y = src_pt[1].y; y>src_pt[0].y; y--) {
                        uchar* fitting_data = src_img.ptr<uchar>(y);
                        for(int x = src_pt[0].x; x<src_pt[1].x; x++) {                                
                                if(fitting_data[x]!= (uchar)0) {
                                        if(x == src_pt[0].x){
                                                int my_sum = 0;
                                                for(int i = x; i<x+10; i++){
                                                        if(fitting_data[i] != (uchar)0){
                                                                my_sum++;
                                                        }
                                                }
                                                if(my_sum > 5){
                                                       dst.push_back(cv::Point(x,y));
                                                       if(dst.size()>20){
                                                               return true; 
                                                       }        
                                                }
                                        }
                                        dst.push_back(cv::Point(x,y));
                                        break;
                                }
                        }
                }
                if(dst.size()<10) return_val = false;
                //if(!dst.empty()){
                        // for(uint i = 0; i<dst.size()-1; i++){
                        //         if(abs(dst[i].x - dst[i+1].x)>10){
                        //                 if((i+6) < dst.size()){
                        //                         int noise = 0;
                        //                         std::vector<int> noise_index;
                        //                         for(uint j = i+2; j<i+7; j++){
                        //                                 if(abs(dst[i].x-dst[j].x)>10){
                        //                                         noise++;
                        //                                         noise_index.push_back(j);   
                        //                                 }
                        //                         }
                        //                         if(noise>2){
                        //                                 uchar* delete_data = src_img.ptr<uchar>(dst[i].y);
                        //                                 delete_data[dst[i].x] = 0;
                        //                                 dst.erase(dst.begin()+i); 
                        //                         }
                        //                         else{
                        //                                 uchar* delete_data = src_img.ptr<uchar>(dst[noise_index[0]].y);
                        //                                 delete_data[dst[noise_index[0]].x] = 0;
                        //                                 delete_data = src_img.ptr<uchar>(dst[noise_index[1]].y);
                        //                                 delete_data[dst[noise_index[1]].x] = 0;
                        //                                 dst.erase(dst.begin()+noise_index[0]);
                        //                                 dst.erase(dst.begin()+noise_index[1]);
                        //                         }
                        //                 }
                        //         }   
                        // }  
                //}
                
        }
        if(lane_name == "dot_test"){
                //int check = 0;
                std::vector<int> lane_width;
                int my_sum = 0, my_avg = 0, my_cnt = 0;
                //saving lane width & inner lane point 
                for(int y = src_pt[1].y; y>src_pt[0].y; y--) {
                        uchar* fitting_data = src_img.ptr<uchar>(y);
                        for(int x = src_pt[0].x; x<src_pt[1].x; x++) {                                
                                if(fitting_data[x]!= (uchar)0) {
                                        dst.push_back(cv::Point(x,y));
                                        int i = x, width_sum = 0, no_point = 0;
                                        while(no_point < 4){   
                                                if(i >= src_pt[1].x-1){
                                                        lane_width.push_back(width_sum);
                                                        my_sum += width_sum;
                                                        break;
                                                }
                                                if(fitting_data[i] != (uchar)0){
                                                        no_point = 0;
                                                        width_sum++;
                                                        i++;
                                                }
                                                else{
                                                        no_point++;
                                                        width_sum++;
                                                        i++;
                                                }
                                        }
                                        if(width_sum > 3){
                                                lane_width.push_back(width_sum);
                                                my_sum += width_sum;
                                                my_cnt++;
                                        }
                                        break;
                                }
                                
                        }
                }
                
                if(!lane_width.empty()){
                        
                        my_avg = my_sum/lane_width.size(); 
                       
                        //check lane slope (parking dot lane slope is lete slope)
                        int change_check_y = 0, l_slope = 0, r_slope = 0;
                        for(uint i = 0; i<dst.size(); i++){
                                //** check dot lane
                                if(dst[i].x >= dst[i+1].x){
                                        l_slope++;
                                }
                                else{
                                        r_slope++;
                                }
                                
                        }
                        std::vector<cv::Point> dot_interval;
                        if(l_slope >= r_slope){
                                //check lane interval(dot lane characteristic is regular interval)
                                for(uint i = 0; i<dst.size()-1; i++){
                                        if(abs(dst[i].y - dst[i+1].y)>6){
                                                
                                                change_check_y++;
                                                //std::cout<<"                            dst["<<i<<"] : "<<dst[i]<<", ";
                                                //std::cout<<"dst["<<i+1<<"] : "<<dst[i+1]<<std::endl;
                                        } 
                                }
                        }
                        else{
                                return_val = false;
                        }
                        if(change_check_y >= 3){
                                //checking dot lane width is reliable
                                int reliability = 0;
                                for(int i = 0; i<lane_width.size(); i++){
                                        //if(my_avg > 10){
                                                if(abs(lane_width[i] - my_avg) < 8){
                                                        reliability++;
                                                }
                                        //}
                                }
                                if(reliability > lane_width.size()*0.4){
                                        return_val = true;
                                }
                                else{
                                        return_val = false;
                                }
                        }
                        else{
                                return_val = false;
                        }   
                }
                else{
                        return_val = false;
                }
                if(return_val == false && !dst.empty()){
                        dst.clear();
                        dst.resize(0);
                }
                      
        }
        return return_val; 
}
 int InitImgObjectforROS::FittedLaneCheck(std::vector<cv::Point> src, const string &lane_name){
        int turn_value = -999;
        if(lane_name == "left"){
                int score = 0;//, steep_slope = 0;
                uint steep_state1 = (src.size()/4)*1, steep_state2 = (src.size()/4)*2, steep_state3 = (src.size()/4)*3;//, steep_state4 = (src.size()/4)*4;
                float steep_diff1 = -1, steep_diff2 = -1, steep_diff3 = -1;
                for(uint i = 0; i<src.size(); i++){
                        if(src[i].x > src[i+1].x){
                                score--;//weighted turn left
                        }
                        else{
                                score++;//weighted turn right(직선 노란차선은 보통 right 가중이 기본)
                        }
                        if(i == steep_state1){//diff<0 : 우회전, diff>0 : 좌회전 
                                steep_diff1 = atan2f((float)src[steep_state1].y - (float)src[0].y,(float)src[steep_state1].x - (float)src[0].x);
                                steep_diff1 = steep_diff1 * 180.0 / CV_PI;
                                if(steep_diff1 > 9999) steep_diff1 = 1000;
                                if(steep_diff1 < -9999) steep_diff1 = -1000;
                        }
                        else if(i == steep_state2){
                                steep_diff2 = atan2f((float)src[steep_state2].y - (float)src[steep_state1].y, (float)src[steep_state2].x - (float)src[steep_state1].x);
                                steep_diff2 = steep_diff2 * 180.0 / CV_PI;
                                if(steep_diff2 > 9999) steep_diff2 = 1000;
                                if(steep_diff2 < -9999) steep_diff2 = -1000;
                        }
                        else if(i == steep_state3){
                                steep_diff3 = atan2f((float)src[steep_state3].y - (float)src[steep_state2].y, (float)src[steep_state3].x - (float)src[steep_state2].x);
                                steep_diff3 = steep_diff3 * 180.0 / CV_PI;
                                if(steep_diff3 > 9999) steep_diff3 = 1000;
                                if(steep_diff3 < -9999) steep_diff3 = -1000;
                        }
                }
                if(steep_diff1 != -1){
                        left_doubling_vec.push_back(steep_diff1);
                }
                
                // std::cout.precision(3);
                // std::cout<<"steep_state2 - steep_state1(left): " <<steep_diff1<<std::endl;
                // std::cout.precision(3);
                // std::cout<<"steep_state3 - steep_state2(left) : " <<steep_diff2<<std::endl;
                // std::cout.precision(3);
                // std::cout<<"steep_state3 - steep_state2(left) : " <<steep_diff3<<std::endl;
                // std::cout<<"avg : "<<(steep_diff1+steep_diff2+steep_diff3)/3<<std::endl;
                if(score > 0) {
//                        std::cout<<"turn right(by left lane)"<<std::endl;
                        turn_value = 1;
                        // if(abs(steep_diff1 - steep_diff2) > 10){
                        //         if(abs(steep_diff1 - ))
                        // }
                }
                else {
                        int change_check_x = -1, max_x = -9999, min_x = 9999, max_x_tmp = 0, min_x_tmp = 0, max_loop = -1, min_loop = -1;
                        //int change_check_y = 0, num_y = 0, pre_num_y = 0;
                        for(uint i = 0; i<src.size(); i++){
                                //** check zigzag
                                if(src[i].x > src[i+1].x){
                                        if(min_loop == 0) change_check_x++;
                                        max_x_tmp = src[i].x;
                                        max_loop = 0;
                                        if(max_x_tmp >= max_x){
                                                max_x = max_x_tmp;
                                        }
                                        min_loop = 1;
                                }
                                else{
                                        if(max_loop == 0) change_check_x++;
                                        min_loop = 0;
                                        min_x_tmp = src[i].x;
                                        if(min_x_tmp <= min_x){
                                                min_x = min_x_tmp;
                                        }
                                        max_loop = 1;
                                }
                                
                        }
                        if(change_check_x >=3 && max_x_tmp - min_x_tmp > 20) {
                                //std::cout<<"ZigZag!!~~(left)"<<std::endl;//right만 지그재그일경우 거긴 점선임
                                turn_value = 0;
                        }
                        else{
                                //std::cout<<"============turn left(by left lane)"<<std::endl;
                                turn_value = -1;
                        }
                        
                }

        }

        if(lane_name == "right"){
                int score = 0;// steep_slope = 0;
                uint steep_state1 = (src.size()/4)*1, steep_state2 = (src.size()/4)*2, steep_state3 = (src.size()/4)*3, steep_state4 = (src.size()/4)*4;
                float steep_diff1 = -1, steep_diff2 = -1, steep_diff3 = -1;
                for(uint i = 0; i<src.size(); i++){
                        if(src[i].x >= src[i+1].x){
                                score--;//weighted turn left(직선 흰차선은 보통 left 가중이 기본)
                        }
                        else{
                                score++;//weighted turn right
                        }
                        if(i == steep_state1){//diff<0 : 우회전, diff>0 : 좌회전 
                                steep_diff1 = atan2f((float)src[steep_state1].y - (float)src[0].y,(float)src[steep_state1].x - (float)src[0].x);
                                steep_diff1 = steep_diff1 * 180.0 / CV_PI;
                                if(steep_diff1 > 999) steep_diff1 = 1000;
                                if(steep_diff1 < -999) steep_diff1 = -1000;
                        }
                        else if(i == steep_state2){
                                steep_diff2 = atan2f((float)src[steep_state2].y - (float)src[steep_state1].y,(float)src[steep_state2].x - (float)src[steep_state1].x);
                                steep_diff2 = steep_diff2 * 180.0 / CV_PI;
                                if(steep_diff2 > 999) steep_diff2 = 1000;
                                if(steep_diff2 < -999) steep_diff2 = -1000;
                                
                        }
                        else if(i == steep_state3){
                                steep_diff3 = atan2f((float)src[steep_state4].y - (float)src[steep_state3].y,(float)src[steep_state4].x - (float)src[steep_state3].x);
                                steep_diff3 = steep_diff3 * 180.0 / CV_PI;
                                if(steep_diff3 > 999) steep_diff3 = 1000;
                                if(steep_diff3 < -999) steep_diff3 = -1000;
                        }

                }
                if(steep_diff1 != -1){
                        right_doubling_vec.push_back(steep_diff1);
                }
                
                // std::cout.precision(3);
                // std::cout<<"steep_state2 - steep_state1(left): " <<steep_diff1<<std::endl;
                // std::cout.precision(3);
                // std::cout<<"steep_state3 - steep_state2(left) : " <<steep_diff2<<std::endl;
                // std::cout.precision(3);
                // std::cout<<"steep_state3 - steep_state2(left) : " <<steep_diff3<<std::endl;
                // std::cout.precision(3);
                // std::cout<<"avg : "<<(steep_diff1+steep_diff2+steep_diff3)/3.0<<std::endl;
                if(score > 0) {
                        //std::cout<<"turn right"<<std::endl;
                        turn_value = 1;
                }
                else {
                        int change_check_x = -1, max_x = -9999, min_x = 9999, max_x_tmp = 0, min_x_tmp = 0, max_loop = -1, min_loop = -1;
                        //int change_check_y = 0, num_y = 0, pre_num_y = 0;
                        for(uint i = 0; i<src.size()/2; i++){
                                //** check zigzag
                                if(src[i].x > src[i+1].x){
                                        if(min_loop == 0 && (max_x - min_x > 20)) change_check_x++;
                                        max_x_tmp = src[i].x;
                                        max_loop = 0;
                                        if(max_x_tmp >= max_x){
                                                max_x = max_x_tmp;
                                        }
                                        min_loop = 1;
                                }
                                else{
                                        if(max_loop == 0 && (max_x - min_x > 20)) change_check_x++;
                                        min_loop = 0;
                                        min_x_tmp = src[i].x;
                                        if(min_x_tmp <= min_x){
                                                min_x = min_x_tmp;
                                        }
                                        max_loop = 1;
                                }
                                  
                        }
                        if(change_check_x >=5 && max_x_tmp - min_x_tmp > 20) {
                                //std::cout<<"ZigZag!!~~(right)"<<std::endl;//right만 지그재그일경우 거긴 점선임
                                turn_value = 0;
                        }
                        else{
                                //std::cout<<"============turn left(by right lane)"<<std::endl;
                                turn_value = -1;
                        }       
                }
        }

        if(lane_name == "dot_test"){
                int change_check_y = 0, l_slope = 0, r_slope = 0;
                // for(uint i = 0; i<src.size()/2; i++){
                //         //** check dot lane
                //         pre_num_y = num_y;
                //         if(src[i].y > src[i+1].y){
                //                 num_y++;
                //         }
                //         else{
                //                 num_y--;
                //         }
                //         if((pre_num_y > num_y && abs(src[i].y - src[i+1].y)>6) ||
                //            (pre_num_y < num_y && abs(src[i].y - src[i+1].y)>6)) {
                //                 change_check_y++;
                //         }  
                // }
                for(uint i = 0; i<src.size(); i++){
                        //** check dot lane
                        if(src[i].x >= src[i+1].x){
                                l_slope++;
                        }
                        else{
                                r_slope++;
                        }
                }
                std::vector<cv::Point> dot_interval;
                if(l_slope >= r_slope){
                        
                        for(uint i = 0; i<src.size(); i++){
                              if(abs(src[i].y - src[i+1].y)>6){
                                      dot_interval.push_back(src[i]);
                                      dot_interval.push_back(src[i+1]);
                                      change_check_y++;
                              } 
                        }
                }  
                if(change_check_y >= 3) {
                        
                        //std::cout<<"#########################################dot lane!!~~(right)"<<std::endl;//right만 지그재그일경우 거긴 점선임
                        //std::cout<<"dot : "<< src<<std::endl;
                        turn_value = 3;
                }
        }
        return turn_value;
 }

void InitImgObjectforROS::setMyCannywithROI(const string &lane_name, cv::Mat &dst){
        //dst = src.clone();
        cv::cvtColor(dst,dst,CV_BGR2GRAY);   
        cv::Canny(dst, dst, (dst.rows + dst.cols) / 4, (dst.rows + dst.cols) / 2);
        cv::medianBlur(dst, dst, 1);
        if(lane_name == "left"){
                for(int y = 0; y<dst.rows/2 ; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x<dst.cols; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for(int y = dst.rows-3; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x<dst.cols; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for(int y = dst.rows/2; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = dst.cols/2; x<dst.cols; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for(int y = dst.rows/2 ; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x<11; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                
        }
        else if(lane_name == "white"){
                for(int y = 0; y<dst.rows/2 ; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x<dst.cols; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for(int y = dst.rows/2 ; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = 0; x < dst.cols/2; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for(int y = dst.rows/2 ; y<dst.rows; y++){
                        uchar* none_roi_data = dst.ptr<uchar>(y);
                        for(int x = dst.cols-10; x<dst.cols-1; x++){
                                if(none_roi_data[x] != (uchar)0){
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
        }
        
}

void InitImgObjectforROS::errorLaneCheck(const string &lane_name, cv::Mat& src, std::vector<cv::Point> &lane_data, std::vector<cv::Point> lane_roi){
        if(lane_name == "left"){
                if(lane_data.size()<30){
                        lane_data.clear();
                        lane_data.resize(0);
                        src = cv::Mat::zeros(src.size(), CV_8UC1);
                }
                else{
                        int error_roi = 0;
                        for(int i = 0; i<20; i++){
                                if(lane_data[i].y < (lane_roi[0].y+lane_roi[1].y)/2){
                                        error_roi++;
                                }
                        }
                        int l_slope = 0, l_temp = 0;
                        int r_slope = 0, r_temp = 0;
                        if(error_roi >= 10){
                                for(uint i = 0; i<lane_data.size()-1; i++){
                                        if(i%5 == 0){
                                                if(r_temp > l_temp){
                                                        r_slope++;
                                                }
                                                else{
                                                        l_slope++;
                                                }
                                                r_temp = 0;
                                                l_temp = 0;
                                        }
                                        if(lane_data[i].x > lane_data[i+1].x){
                                                r_temp++;
                                        }
                                        else{
                                                l_temp++;
                                        }
                                }
                                if(r_slope < l_slope){
                                        lane_data.clear();
                                        lane_data.resize(0);
                                        src = cv::Mat::zeros(src.size(), CV_8UC1);  
                                }
                        }
                }
        }
        else if(lane_name == "right"){
                if(lane_data.size()<30){
                        lane_data.clear();
                        lane_data.resize(0);
                        src = cv::Mat::zeros(src.size(), CV_8UC1);
                }
                else{
                        int error_roi = 0;
                        for(int i = 0; i<20; i++){
                                if(lane_data[i].y < (lane_roi[0].y+lane_roi[1].y)/2){
                                        error_roi++;
                                }
                        }
                        int l_slope = 0, l_temp = 0;
                        int r_slope = 0, r_temp = 0;
                        if(error_roi >= 10){
                                for(uint i = 0; i<lane_data.size()-1; i++){
                                        if(i%5 == 0){
                                                if(r_temp > l_temp){
                                                        r_slope++;
                                                }
                                                else{
                                                        l_slope++;
                                                }
                                                r_temp = 0;
                                                l_temp = 0;
                                        }
                                        if(lane_data[i].x > lane_data[i+1].x){
                                                r_temp++;
                                        }
                                        else{
                                                l_temp++;
                                        }
                                }
                                if(r_slope < l_slope){
                                        lane_data.clear();
                                        lane_data.resize(0);
                                       // src = cv::Mat::zeros(src.size(), CV_8UC1);  
                                }
                        }
                }
        }
}

void InitImgObjectforROS::errorLaneCheck2(cv::Mat& src, std::vector<cv::Point> &left_lane_data, std::vector<cv::Point> &right_lane_data){
       
                // if(lane_data.size()<30){
                //         lane_data.clear();
                //         lane_data.resize(0);
                //         src = cv::Mat::zeros(src.size(), CV_8UC1);
                // }
                // else{
                //         int error_roi = 0;
                //         for(int i = 0; i<20; i++){
                //                 if(lane_data[i].y < (lane_roi[0].y+lane_roi[1].y)/2){
                //                         error_roi++;
                //                 }
                //         }
                //         int l_slope = 0, l_temp = 0;
                //         int r_slope = 0, r_temp = 0;
                //         if(error_roi >= 10){
                //                 for(uint i = 0; i<lane_data.size()-1; i++){
                //                         if(i%5 == 0){
                //                                 if(r_temp > l_temp){
                //                                         r_slope++;
                //                                 }
                //                                 else{
                //                                         l_slope++;
                //                                 }
                //                                 r_temp = 0;
                //                                 l_temp = 0;
                //                         }
                //                         if(lane_data[i].x > lane_data[i+1].x){
                //                                 r_temp++;
                //                         }
                //                         else{
                //                                 l_temp++;
                //                         }
                //                 }
                //                 if(r_slope < l_slope){
                //                         lane_data.clear();
                //                         lane_data.resize(0);
                //                         src = cv::Mat::zeros(src.size(), CV_8UC1);  
                //                 }
                //         }
                // }
}
int InitImgObjectforROS::slopeCheck(cv::Mat &src, std::vector<cv::Point> lane_data, const string &lane_name){
        cv::Point center_left = cv::Point(src.cols/2-50, src.rows/2);
        cv::Point center_right = cv::Point(src.cols/2+50, src.rows/2);
        cv::Point center = cv::Point(src.cols/2, src.rows/2);
        if(lane_name == "left"){
                bool check1 = false, check2 = false, check3 = false;
                int interval1 = -1, interval2 = -1, interval3 = -1;
                 
                for(int i = 0; i<lane_data.size(); i++){
                        if(lane_data[i].y >= 180 && lane_data[i].y < 190 && !check1){
                                cv::line(src,cv::Point(center.x,lane_data[i].y),cv::Point(lane_data[i].x,lane_data[i].y),cv::Scalar(200,222,23),2);
                                interval1 = abs(center.x - lane_data[i].x);
                                check1 = true;
                        }
                        else if(lane_data[i].y >= 170 && lane_data[i].y < 180 && !check2){ 
                                cv::line(src,cv::Point(center.x,lane_data[i].y),cv::Point(lane_data[i].x,lane_data[i].y),cv::Scalar(200,222,23),2);
                                interval2 = abs(center.x - lane_data[i].x);
                                check2 = true;
                        }
                        else if(lane_data[i].y >= 160 && lane_data[i].y < 170 && !check3){
                                cv::line(src,cv::Point(center.x,lane_data[i].y),cv::Point(lane_data[i].x,lane_data[i].y),cv::Scalar(200,222,23),2);
                                interval3 = abs(center.x - lane_data[i].x);
                                check3 = true;
                        }
                        
                }
                if(interval1 <10 && interval2 < 10 && interval3 < 10){
                        return -999;
                }
                if(interval1 >= interval2 && interval2 >= interval3){
                        if(interval1 - interval2 < 8 && interval2 - interval3 <8){
                                cv::putText(src, "go staight", center_left, 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(5,25,255), 2);
                                return 0;
                        }
                        else{
                                cv::putText(src, "turn right", center_left, 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(5,25,255), 2);
                                return 1;
                        }
                }
                else if(interval1 < interval2 && interval2 < interval3){
                        cv::putText(src, "turn left", center_left, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(5,25,255), 2);
                        return -1;
                }
                else{
                        return -999;
                }
                // else if(interval1>interval2 && interval3>interval2){
                //         cv::putText(src, "turn left", center_left, 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(5,25,255), 2);
                //         return -1;
                // }
                // else if(interval1<interval2 && interval3<interval2){
                //         cv::putText(src, "turn right", center_left, 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(5,25,255), 2);
                //         return 1;
                // }
        }
        if(lane_name == "right"){
                bool check1 = false, check2 = false, check3 = false;
                int interval1 = -1, interval2 = -1, interval3 = -1;
                 
                for(int i = 0; i<lane_data.size(); i++){
                        if(lane_data[i].y >= 180 && lane_data[i].y < 190 && !check1){
                                cv::line(src,cv::Point(center.x,lane_data[i].y),cv::Point(lane_data[i].x,lane_data[i].y),cv::Scalar(200,222,23),2);
                                interval1 = abs(center.x - lane_data[i].x);
                                check1 = true;
                        }
                        else if(lane_data[i].y >= 170 && lane_data[i].y < 180 && !check2){ 
                                cv::line(src,cv::Point(center.x,lane_data[i].y),cv::Point(lane_data[i].x,lane_data[i].y),cv::Scalar(200,222,23),2);
                                interval2 = abs(center.x - lane_data[i].x);
                                check2 = true;
                        }
                        else if(lane_data[i].y >= 160 && lane_data[i].y < 170 && !check3){
                                cv::line(src,cv::Point(center.x,lane_data[i].y),cv::Point(lane_data[i].x,lane_data[i].y),cv::Scalar(200,222,23),2);
                                interval3 = abs(center.x - lane_data[i].x);
                                check3 = true;
                        }
                        
                }
                if(interval1 <10 && interval2 < 10 && interval3 < 10){
                        return -999;
                }
                if(interval1 > interval2 && interval2 > interval3){
                        cv::putText(src, "turn left", center_right, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,252,155), 2);
                        cv::putText(src, std::to_string(interval1), cv::Point(src.cols/2 + 40, src.rows/2+100), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        cv::putText(src, std::to_string(interval2), cv::Point(src.cols/2 + 40, src.rows/2+70), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        cv::putText(src, std::to_string(interval3), cv::Point(src.cols/2 + 40, src.rows/2+30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        return 1;
                }
                else if(interval1 <= interval2 && interval2 <= interval3){
                        if(interval1 - interval2 < 10 && interval2 - interval3 <10){
                                cv::putText(src, "go staight", center_right, 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(5,25,155), 2);
                                cv::putText(src, std::to_string(interval1), cv::Point(src.cols/2 + 40, src.rows/2+100), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        cv::putText(src, std::to_string(interval2), cv::Point(src.cols/2 + 40, src.rows/2+70), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        cv::putText(src, std::to_string(interval3), cv::Point(src.cols/2 + 40, src.rows/2+30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                                return 0;
                        }
                        else{
                                cv::putText(src, "turn right", center_right, 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,25,155), 2);
                                cv::putText(src, std::to_string(interval1), cv::Point(src.cols/2 + 40, src.rows/2+100), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        cv::putText(src, std::to_string(interval2), cv::Point(src.cols/2 + 40, src.rows/2+70), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                        cv::putText(src, std::to_string(interval3), cv::Point(src.cols/2 + 40, src.rows/2+30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                                return -1;
                        }
                }
                else{
                        return -999;
                }
                // else if(interval1>interval2 && interval3>interval2){
                //         cv::putText(src, "turn left", center_right, 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,252,155), 2);
                //         cv::putText(src, std::to_string(interval1), cv::Point(src.cols/2 + 40, src.rows/2+100), 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                //         cv::putText(src, std::to_string(interval2), cv::Point(src.cols/2 + 40, src.rows/2+70), 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                //         cv::putText(src, std::to_string(interval3), cv::Point(src.cols/2 + 40, src.rows/2+30), 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                //         return -1;
                // }
                // else if(interval1<interval2 && interval3<interval2){
                //         cv::putText(src, "turn right", center_right, 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,252,155), 2);
                //         cv::putText(src, std::to_string(interval1), cv::Point(src.cols/2 + 40, src.rows/2+100), 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                //         cv::putText(src, std::to_string(interval2), cv::Point(src.cols/2 + 40, src.rows/2+70), 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                //         cv::putText(src, std::to_string(interval3), cv::Point(src.cols/2 + 40, src.rows/2+30), 
                //         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50,252,155), 1);
                //         return 1;
                // /}
        }
        
}

void InitImgObjectforROS::calSlope(cv::Mat &src, std::vector<cv::Point> lane_data, const string &lane_name){
        int lane_size = 30;
        if(lane_data.size() < 30) lane_size = lane_data.size();
        if(lane_name == "left"){
                
                double xsum=0,x2sum=0,ysum=0,xysum=0;//variables for sums/sigma of xi,yi,xi^2,xiyi etc
                for (int i=0;i<lane_size;i++){
                        xsum = xsum + lane_data[i].x;              //calculate sigma(xi)
                        ysum = ysum + lane_data[i].y;              //calculate sigma(yi)
                        x2sum = x2sum + pow(lane_data[i].x,2);     //calculate sigma(x^2i)
                        xysum = xysum + lane_data[i].x * lane_data[i].y;       //calculate sigma(xi*yi)
                }
                double a = 0, b = 0;
                a = ((lane_size*xysum) - xsum*ysum)/((lane_size * x2sum) - (xsum*xsum));            //calculate slope
                b = ((x2sum*ysum) - (xsum*xysum))/((x2sum*lane_size) - (xsum*xsum));            //calculate intercept
                // double y_fit[lane_data.size()];                        //an array to store the new fitted values of y    
                // for (i=0;i<lane_data.size();i++){
                //         y_fit[i]=a*x[i]+b;   //to calculate y(fitted) at given x points
                // }
                cout<<"\nThe linear fit line is of the form:\n\n"<<a<<"x + "<<b<<endl;        //print the best fit line
                left_slope = a;
        }
        else if(lane_name == "right"){
                double xsum=0,x2sum=0,ysum=0,xysum=0;//variables for sums/sigma of xi,yi,xi^2,xiyi etc
                for (int i=0;i<lane_size;i++){
                        xsum = xsum + lane_data[i].x;              //calculate sigma(xi)
                        ysum = ysum + lane_data[i].y;              //calculate sigma(yi)
                        x2sum = x2sum + pow(lane_data[i].x,2);     //calculate sigma(x^2i)
                        xysum = xysum + lane_data[i].x * lane_data[i].y;       //calculate sigma(xi*yi)
                }
                double a = 0, b = 0;
                a = ((lane_size*xysum) - xsum*ysum)/((lane_size * x2sum) - (xsum*xsum));            //calculate slope
                b = ((x2sum*ysum) - (xsum*xysum))/((x2sum*lane_size) - (xsum*xsum));            //calculate intercept
                // double y_fit[lane_data.size()];                        //an array to store the new fitted values of y    
                // for (i=0;i<lane_data.size();i++){
                //         y_fit[i]=a*x[i]+b;   //to calculate y(fitted) at given x points
                // }
                cout<<"\nThe linear fit line is of the form:\n\n"<<a<<"x + "<<b<<endl;        //print the best fit line
                right_slope = a;
        }
        
        
}

bool InitImgObjectforROS::detectBlockingBar(cv::Mat src){
        int first_red = -1, last_red = -1, red_sum = 0, no_pt = 0, red_cnt = 0, stop = 0;
        for(int y = 0; y<100; y++){//blockbar roi
                uchar* red_data = src.ptr<uchar>(y);
                no_pt = 0;
                for(int x = 0; x<src.cols; x++){
                        if(red_data[x] != 0){
                                if(no_pt <= 5) no_pt = 0;
                                red_sum++;
                                if(red_sum > 20){
                                        red_sum = 0;
                                        no_pt = 0;
                                        for(int i = x+1; i<src.cols; i++){
                                                if(red_data[i] != 0){
                                                        red_sum++;
                                                        if(red_sum > 20 && no_pt > 20){
                                                                red_cnt++;
                                                                red_sum = 0;
                                                                no_pt = 0;
                                                                if(red_cnt > 2) {
                                                                        stop = 1;//detect blocking bar
                                                                        break;
                                                                }
                                                        }
                                                }
                                                else{ 
                                                        no_pt++;
                                                }
                                                
                                        }
                                        break;
                                }
                        }
                        else{
                                no_pt++;
                                if(no_pt > 5) break;
                        }
                }
                if(stop == 1 ) break;
        }
        if(stop == 1) return true;
        else return false;
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
                        video_main.open("/home/seuleee/Desktop/autorace_video_src/0810/main_record.avi",cv::VideoWriter::fourcc('X','V','I','D'),fps,cv::Size(640/2,480/2), isColor);
        }
        
        while(img_obj.nh.ok()) {
                img_obj.pub.publish(img_obj.coordi_array);
                img_obj.ang_vel_pub.publish(img_obj.goal_array);
               // if(img_obj.rightlane_turn_value_num == 30 && img_obj.leftlane_turn_value_num == 30){
                        img_obj.goal_pub.publish(img_obj.goal_array);
               // }
                img_obj.reset_msg_pub.publish(img_obj.reset_val);
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

