#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <lane_detection_func/lane_detection_func.hpp>
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Bool.h"
#include <time.h>
#include <string>
#include <math.h>

/////////mission check///////
bool parking_mode = false, parking_checked = false;
int parking_reliabilty = 0, go_cnt = 0;
cv::Point first_point;
int parking_stage = 0;
bool blocking_bar_mode = false, blocking_bar_checked = false;
int blocking_bar_reliabilty = 0, blocking_bar_stage = 0, blocking_bar_first = 0;
bool tunnel_mode = false, tunnel_checked = false;
int tunnel_reliabilty = 0;
bool signal_lamp_mode = false, signal_lamp_checked = false;
int signal_lamp_stage = 0, red_rotation = 0, yellow_rotation = 0;
int red_reliabilty = 0, green_reliabilty = 0, yellow_reliabilty = 0;
bool normal_mode = true;
int before_signal = -1;
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

static int for_gui;
static int gazebo;
static int bird_eye_view;
static int auto_shot;
static int auto_record;

static int left_min_interval, left_max_interval;
static int right_min_interval, right_max_interval;

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

//for street sign
static int b_hmin, b_hmax, b_smin, b_smax, b_vmin, b_vmax;

static int reset_msg;

double fps = 5;
int fourcc = CV_FOURCC('X', 'V', 'I', 'D'); // codec
bool isColor = true;
cv::Point center_pt_t, center_pt_b;

cv::VideoWriter video_left;
cv::VideoWriter video_right;
cv::VideoWriter video_main;

static std::string groupName;

int dot_cnt = 0;

int pre_left_interval = -2, pre_right_interval = -2;
int left_interval = -1, right_interval = -1;
int pre_left_size = -1, pre_right_size = -1;

std::vector<int> left_interval_vec, right_interval_vec;

double left_roi_slope = -999.9, right_roi_slope = -999.9;
double pre_left_roi_slope = -999.9, pre_right_roi_slope = -999.9;

int left_center_pt = -1, right_center_pt = -1;
int l_r_center_pt = -1, img_center_left_pt = 145, img_center_right_pt = 175;
float x_goal_ = 0, y_goal_ = 0, prev_y_goal_ = 0;

cv::Mat left_fit_img, right_fit_img;

static int motor_feq = 0;
using namespace lane_detect_algo;
using namespace std;

class InitImgObjectforROS
{

      public:
        ros::NodeHandle nh;
        image_transport::ImageTransport it;
        image_transport::Subscriber sub_img;

        std_msgs::Int32MultiArray coordi_array;
        std_msgs::Float32MultiArray goal_array;
        std_msgs::Bool reset_val;
        ros::Publisher pub = nh.advertise<std_msgs::Int32MultiArray>("/" + groupName + "/lane", 100); //Topic publishing at each camera
        ros::Publisher goal_pub = nh.advertise<std_msgs::Float32MultiArray>("/" + groupName + "/pixel_goal", 100);
        ros::Publisher ang_vel_pub = nh.advertise<std_msgs::Float32MultiArray>("/" + groupName + "/angular_vel", 100);
        ros::Publisher reset_msg_pub = nh.advertise<std_msgs::Bool>("/" + groupName + "/reset_msg", 100);
        int imgNum = 0; //for saving img
        int msg_count_left = 0, msg_count_right = 0;
        int left_direction_cnt = 0, right_direction_cnt = 0;
        cv::Mat output_origin_for_copy; //for saving img
        InitImgObjectforROS();
        ~InitImgObjectforROS();
        void imgCb(const sensor_msgs::ImageConstPtr &img_msg);
        void initParam();
        void initMyHSVTrackbar(const string &trackbar_name);
        void initMyRESETTrackbar(const string &trackbar_name);
        void setMyHSVTrackbarValue(const string &trackbar_name);
        void setMyRESETTrackbarValue(const string &trackbar_name);
        void setColorPreocessing(lane_detect_algo::CalLane callane, cv::Mat src, cv::Mat &dst_y, cv::Mat &dst_w, cv::Mat &dst_r, cv::Mat &dst_r2, cv::Mat &dst_g, cv::Mat &dst_y2, cv::Mat &dst_b, cv::Mat &parking_white);
        void setProjection(lane_detect_algo::CalLane callane, cv::Mat src, unsigned int *H_aix_Result_color);
        void restoreImgWithLangeMerge(lane_detect_algo::CalLane callane, cv::Mat origin_size_img, cv::Mat src_y, cv::Mat src_w, cv::Mat &dst);
        void extractLanePoint(cv::Mat origin_src, cv::Mat lane_src);

        void setMySobel(cv::Mat &dst);
        void setMyLaneBox(cv::Point t_pt, cv::Point b_pt, const string &lane_name, std::vector<cv::Point> &dst);
        bool setMyLaneFitting(cv::Mat &src_img, std::vector<cv::Point> src_pt, const string &lane_name, std::vector<cv::Point> &dst);
        void setRoi(const string &lane_name, cv::Mat &dst);
        void setRoi(const string &lane_name, cv::Mat &dst, std::vector<cv::Point> &pt_dst);
        void setMyCanny(const string &lane_name, cv::Mat &dst);
        void setMyMorphology(cv::Mat &dst);

        int checkDirectionWithFlann(cv::Mat &src, int min_hessian);
        bool checkParkingWithFlann(cv::Mat &src, int min_hessian);
        bool checkTunnelWithFlann(cv::Mat &src, int min_hessian);
        bool checkBlueArea(cv::Mat &src, cv::Point &left_top_blue, cv::Point &right_bottom_blue);

        bool detectBlockingBar(cv::Mat src);
        bool signalRedDetection(cv::Mat src_red);
        bool signalGreenDetection(cv::Mat src_green);

        bool signalYellowDetection(cv::Mat src_yellow);
};

InitImgObjectforROS::InitImgObjectforROS() : it(nh)
{
        initParam();
        if (!web_cam)
        { //'DEBUG_SW == TURE' means subscribing videofile image
                sub_img = it.subscribe("/" + groupName + "/videofile/image_raw", 1, &InitImgObjectforROS::imgCb, this);
        }
        else
        { //'DEBUG_SW == FALSE' means subscribing webcam image
                if (!gazebo)
                { //use webcam topic
                        sub_img = it.subscribe("/" + groupName + "/raw_image", 1, &InitImgObjectforROS::imgCb, this);
                }
                else
                { //use gazebo topic
                        sub_img = it.subscribe("/camera/image", 1, &InitImgObjectforROS::imgCb, this);
                }
        }

        if (track_bar)
        {
                initMyHSVTrackbar(groupName + "_YELLOW_LANE_TRACKBAR");
                initMyHSVTrackbar(groupName + "_WHITE_LANE_TRACKBAR");
                initMyHSVTrackbar(groupName + "_BLOCKING_RED_TRACKBAR");
                initMyHSVTrackbar(groupName + "_SIGNAL_RED_TRACKBAR");
                initMyHSVTrackbar(groupName + "_SIGNAL_GREEN_TRACKBAR");
                initMyHSVTrackbar(groupName + "_SIGNAL_YELLOW_TRACKBAR");
                initMyHSVTrackbar(groupName + "_BLUE_AREA_TRACKBAR");
                initMyRESETTrackbar("reset msg");
        }
}

InitImgObjectforROS::~InitImgObjectforROS()
{
        if (debug)
        { //'DEBUG_SW == TURE' means subscribing videofile image
                cv::destroyWindow(OPENCV_WINDOW_VF);
        }
        else
        { //'DEBUG_SW == FALE' means subscribing webcam image
                cv::destroyWindow(OPENCV_WINDOW_WC);
        }
}

void InitImgObjectforROS::imgCb(const sensor_msgs::ImageConstPtr &img_msg)
{
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat frame, yellow_hsv, white_hsv, origin_white_hsv, origin_yellow_hsv, yellow_labeling, white_labeling;
        cv::Mat yellow_roi, white_roi, white_sobel, yellow_canny;
        cv::Mat laneColor, origin, mergelane;
        cv::Mat red_hsv, red2_hsv, green_hsv, yellow2_hsv, blue_hsv, parking_white, park_origin;
        cv::Mat gui_test;
        std::vector<cv::Point> box_pt_y, box_pt_w;
        std::vector<cv::Point> left_lane_fitting, right_lane_fitting, dot_lane_fitting;
        uint frame_height, frame_width;

        try
        {
                cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
                frame = cv_ptr->image;
                origin = cv_ptr->image;
                if (!frame.empty())
                {
                        if (!gazebo)
                        {                                                                                                   //if you use gazebo topic than this condition not used.
                                cv::resize(origin, frame, cv::Size(origin.cols / 2, origin.rows / 2), 0, 0, CV_INTER_AREA); //img downsizing //320 240
                                /*another solution->*/                                                                      //cv::resize(frame, frame, cv::Size(), 0.2, 0.2 320 240);
                        }
                        if (auto_record)
                        {
                                if (groupName == "left")
                                        video_left << frame;
                                else if (groupName == "right")
                                        video_right << frame;
                                else if (groupName == "main")
                                        video_main << frame;
                        }

                        frame_height = (uint)frame.rows;
                        frame_width = (uint)frame.cols;

                        center_pt_t = cv::Point(frame_width / 2, 0);
                        center_pt_b = cv::Point(frame_width / 2, frame_height - 1);

                        gui_test = frame.clone();
                        parking_white = frame.clone();
                        park_origin = frame.clone();
                        setMyMorphology(parking_white);
                        ////*for lane_detect_func object*////
                        lane_detect_algo::CalLane callane;

                        ////*reset motor trackbar set*////
                        if (track_bar)
                        {
                                cv::Mat reset_img = cv::Mat::zeros(frame.size(), CV_8UC3);
                                setMyRESETTrackbarValue("reset msg");
                                //cv::imshow("motor reset msg trackbar", reset_img);
                                reset_val.data = reset_msg;
                        }

                                             
                        

                        
                        

                        ////*Process color detection including trackbar setting*////
                        setColorPreocessing(callane, frame, yellow_hsv, white_hsv, red_hsv, red2_hsv, green_hsv, yellow2_hsv, blue_hsv, parking_white);
                        origin_white_hsv = white_hsv.clone();
                        origin_yellow_hsv = yellow_hsv.clone();

                       
                        
                        cv::Point left_top_blue, right_bottom_blue;
                        bool is_blue = false;
                        is_blue = checkBlueArea(blue_hsv,left_top_blue,right_bottom_blue);
                        if(is_blue){
                                
                                if(for_gui){
                                        cv::rectangle(gui_test, left_top_blue, right_bottom_blue, cv::Scalar(0, 0, 255), 1);
                                }
                                cv::Rect2d blue_rect = cv::Rect(left_top_blue,right_bottom_blue);
                                cv::Mat blue_src = frame(blue_rect);
                               
                                cv::imshow("vvv",blue_src);

                                //각 단계 진행 이후 유사도 비교 x   
                                checkDirectionWithFlann(blue_src,100);
                                checkParkingWithFlann(blue_src,400);
                        }
                        //노란 레이블 추가
                        checkTunnelWithFlann(frame,400);

                        cv::Mat park_detect = origin_white_hsv.clone();
                        // cv::Sobel(park_detect, park_detect, park_detect.depth(), 0, 1);
                        // cv::medianBlur(park_detect,park_detect,3);
                        std::vector<cv::Point> park_line_test;
                        cv::imshow("park_detect", park_detect);

                        cv::imshow("blocking_bar", red_hsv);
                        ///****canny & sobel***///
                        cv::Mat my_canny = frame.clone();
                        //setRoi("left",my_canny);
                        setMyCanny("left_right", my_canny);
                        cv::imshow("left_right", my_canny);
                        //cv::imshow("canny_left",yellow_canny);
                        //callane.makeContoursLeftLane(yellow_canny, yellow_canny);

                        white_sobel = frame.clone();
                        //setRoi("right",white_sobel);
                        setMySobel(white_sobel);
                        cv::imshow("sobel_right", white_sobel);

                        yellow_labeling = yellow_hsv.clone();
                        //yellow_labeling = yellow_hsv & my_canny;
                        setRoi("left", yellow_labeling, box_pt_y);
                        setMyMorphology(yellow_labeling);
                        //**If you use vector type variable, Please cheack your variable is not empty!
                        left_fit_img = cv::Mat::zeros(yellow_labeling.size(), CV_8UC1);
                        if (!setMyLaneFitting(yellow_labeling, box_pt_y, "left", left_lane_fitting))
                        {
                                left_lane_fitting.clear();
                                left_lane_fitting.resize(0);
                                yellow_labeling = cv::Mat::zeros(yellow_labeling.size(), CV_8UC1);
                                left_interval = -1;
                                left_center_pt = -1;
                        }
                        else
                        {
                                if (!left_lane_fitting.empty())
                                {
                                        if (for_gui)
                                        {
                                                if (left_lane_fitting[0].x != 0)
                                                {
                                                        left_center_pt = (left_lane_fitting[left_lane_fitting.size() - 1].x + left_lane_fitting[0].x) / 2;
                                                        //cv::line(gui_test, cv::Point(left_center_pt, 100), cv::Point(left_center_pt, 300), cv::Scalar(23, 32, 100), 2);
                                                }
                                                else
                                                {
                                                        left_center_pt = -1;
                                                }
                                        }

                                        std::vector<cv::Point> left_roi_fitting_vec;

                                        int condition_check_left = 0;
                                        int left_pt_sum = 0, left_pt_avg = 0, left_pt_cnt = 0;
                                        for (int i = 0; i < left_lane_fitting.size(); i++)
                                        {
                                                if (left_lane_fitting[i].y >= 165 && left_lane_fitting[i].y <= 185)
                                                {
                                                        left_roi_fitting_vec.push_back(left_lane_fitting[i]);
                                                        condition_check_left++;
                                                        if (for_gui && condition_check_left == 1) //l_cnt vaule need for only one loop this condition
                                                        {
                                                                cv::polylines(gui_test, left_roi_fitting_vec, 0, cv::Scalar(20, 200, 240), 2);
                                                        }
                                                }
                                        }
                                        if (!left_roi_fitting_vec.empty() && left_roi_fitting_vec[0].x != 0) //first value(in this vec[0]) should be none zero
                                        {
                                                left_roi_slope = ((double)(left_roi_fitting_vec[left_roi_fitting_vec.size() - 1].y - left_roi_fitting_vec[0].y) /
                                                                  (double)(left_roi_fitting_vec[left_roi_fitting_vec.size() - 1].x - left_roi_fitting_vec[0].x));
                                                if (isnan(left_roi_slope) || abs(left_roi_slope) > 2)
                                                {
                                                        if (isnan(left_roi_slope) || left_roi_slope > 0)
                                                        {
                                                                left_roi_slope = 11;
                                                        }
                                                        else
                                                        {
                                                                left_roi_slope = 10;
                                                        }
                                                }
                                        }
                                        else
                                        {
                                                left_roi_slope = 11;
                                        }
                                        left_interval = abs(left_lane_fitting[0].x - center_pt_b.x);
                                }
                                else
                                {
                                        left_interval = -1;
                                        left_center_pt = -1;
                                }
                        }

                        white_labeling = white_hsv.clone();
                        //white_labeling = white_hsv & my_canny;
                        setRoi("right", white_hsv, box_pt_w);
                        callane.makeContoursRightLane(white_hsv, white_labeling);
                        setMyMorphology(white_labeling);
                        //**If you use vector type variable, Please cheack your variable is not empty!
                        //*for inner right lane fitting*//
                        right_fit_img = cv::Mat::zeros(white_labeling.size(), CV_8UC1);
                        if (!setMyLaneFitting(white_labeling, box_pt_w, "right", right_lane_fitting))
                        {
                                right_lane_fitting.clear();
                                right_lane_fitting.resize(0);
                                white_labeling = cv::Mat::zeros(white_labeling.size(), CV_8UC1);
                                right_interval = -1;
                                right_center_pt = -1;
                        }
                        //*for inner left lane fitting*//
                        else
                        {
                                if (!right_lane_fitting.empty())
                                {

                                        if (for_gui)
                                        {
                                                if (right_lane_fitting[0].x != 0)
                                                {
                                                        right_center_pt = (right_lane_fitting[right_lane_fitting.size() - 1].x + right_lane_fitting[0].x) / 2;
                                                        //cv::line(gui_test, cv::Point(right_center_pt, 100), cv::Point(right_center_pt, 300), cv::Scalar(23, 32, 100), 2);
                                                        //cv::line(gui_test, cv::Point(right_lane_fitting[right_lane_fitting.size()-1].x-10, right_lane_fitting[right_lane_fitting.size()-1].y), cv::Point(right_lane_fitting[right_lane_fitting.size()-1].x+10, right_lane_fitting[right_lane_fitting.size()-1].y), cv::Scalar(23, 32, 100), 2);
                                                        //cv::line(gui_test, cv::Point(right_lane_fitting[0].x-10, right_lane_fitting[0].y), cv::Point(right_lane_fitting[0].x+10, right_lane_fitting[0].y), cv::Scalar(23, 32, 100), 2);
                                                }
                                                else
                                                {
                                                        right_center_pt = -1;
                                                }
                                        }
                                        std::vector<cv::Point> right_roi_fitting_vec;

                                        int condition_check_right = 0;
                                        for (int i = 0; i < right_lane_fitting.size(); i++)
                                        {
                                                if (right_lane_fitting[i].y >= 165 && right_lane_fitting[i].y <= 185)
                                                {
                                                        right_roi_fitting_vec.push_back(right_lane_fitting[i]);
                                                        condition_check_right++;
                                                        if (for_gui && condition_check_right == 1) //l_cnt vaule need for only one loop this condition
                                                        {
                                                                cv::polylines(gui_test, right_roi_fitting_vec, 0, cv::Scalar(20, 200, 240), 2);
                                                        }
                                                }
                                        }
                                        if (!right_roi_fitting_vec.empty() && right_roi_fitting_vec[0].x != 0)
                                        {
                                                right_roi_slope = ((double)(right_roi_fitting_vec[right_roi_fitting_vec.size() - 1].y - right_roi_fitting_vec[0].y) /
                                                                   (double)(right_roi_fitting_vec[right_roi_fitting_vec.size() - 1].x - right_roi_fitting_vec[0].x));

                                                if (isnan(right_roi_slope) || abs(right_roi_slope) > 2)
                                                {
                                                        if (isnan(right_roi_slope) || right_roi_slope < 0)
                                                        {
                                                                right_roi_slope = 11;
                                                        }
                                                        else
                                                        {
                                                                right_roi_slope = 10;
                                                        }
                                                }
                                        }
                                        else
                                        {
                                                right_roi_slope = 11;
                                        }
                                        right_interval = abs(right_lane_fitting[0].x - center_pt_b.x);
                                }
                                else
                                {
                                        right_interval = -1;
                                        right_center_pt = -1;
                                }
                        }
                        cv::imshow("left_fit",left_fit_img);
                        cv::imshow("right_fit",right_fit_img);

                        std::vector<cv::Vec4i> lines_left, lines_right;
                        std::vector<cv::Vec4i>::iterator it_left, it_right;

                        cv::Mat hough_test = frame.clone();
                       
                       // cv::HoughLinesP(my_canny,lines_left,1,CV_PI/180.40,90,30);
                        
                        //setMyCanny("right",yellow_labeling);
                        //setMyMorphology(yellow_labeling);
                        //setMyMorphology(yellow_labeling);

                        cv::HoughLinesP(white_labeling,lines_right,1,CV_PI/180.100,90,90);
                        cv::HoughLinesP(yellow_labeling,lines_left,1,CV_PI/180.100,100,90);
                        
                        if(for_gui){
                                cv::line(hough_test,cv::Point(frame.cols/2-20,frame.rows/2),cv::Point(frame.cols/2-20,frame.rows+50),cv::Scalar(220,222,22),2,0);
                                cv::line(hough_test,cv::Point(frame.cols/2+20,frame.rows/2),cv::Point(frame.cols/2+20,frame.rows+50),cv::Scalar(220,222,22),2,0);
                        }
                        
                        
                        if(!lines_left.empty() && !lines_right.empty()){
                                float left_ladian, right_ladian;
                                int left_degree, right_degree;
                                int left_hough_x, right_hough_x;
                                
                                it_left = lines_left.end() - 1;
                                it_right = lines_right.end() - 1;
                                left_ladian = atan2f((*it_left)[3]-(*it_left)[1], (*it_left)[2]-(*it_left)[0]);
                                right_ladian = atan2f((*it_right)[3]-(*it_right)[1], (*it_right)[2]-(*it_right)[0]);
                                left_degree = left_ladian * 180/CV_PI;
                                right_degree = right_ladian * 180/CV_PI;

                                printf("degree(left) : %d\n",&left_degree);
                                printf("degree(right) : %d\n",&right_degree);
                                
                                if(abs(left_degree)>=20 && abs(left_degree)<=80){
                                        if(for_gui){
                                                cv::line(hough_test,cv::Point(((*it_left)[0] + (*it_left)[2])/2,((*it_left)[1] + (*it_left)[3])/2),
                                                cv::Point(((*it_left)[0] + (*it_left)[2])/2,((*it_left)[1] + (*it_left)[3])/2+70),
                                                cv::Scalar(250,200,10),3,0);

                                                cv::line(hough_test,cv::Point((*it_left)[0], (*it_left)[1]),
                                                cv::Point((*it_left)[2],(*it_left)[3]),
                                                cv::Scalar(255,200,20),3,0);
      
                                        }
                                        left_hough_x = ((*it_left)[0] + (*it_left)[2])/2;
                                       
                                }
                                
                                if(abs(right_degree)>=20 && abs(right_degree)<=80){
                                        if(for_gui){
                                                cv::line(hough_test,cv::Point(((*it_right)[0] + (*it_right)[2])/2,((*it_right)[1] + (*it_right)[3])/2),
                                                cv::Point(((*it_right)[0] + (*it_right)[2])/2,((*it_right)[1] + (*it_right)[3])/2+70),
                                                cv::Scalar(25,210,40),3,0);

                                                cv::line(hough_test,cv::Point((*it_right)[0], (*it_right)[1]),
                                                cv::Point((*it_right)[2],(*it_right)[3]),
                                                cv::Scalar(25,220,50),3,0);  
                                        }
                                        right_hough_x = ((*it_right)[0] + (*it_right)[2])/2;
                                      
                                }

                                int cur_center_x;
                                cur_center_x = (right_hough_x+left_hough_x)/2;
                                if(for_gui){
                                        cv::line(hough_test,cv::Point(cur_center_x,hough_test.rows/2),
                                        cv::Point(cur_center_x,hough_test.rows/2 + 70),
                                        cv::Scalar(25,110,40),3,0);
                                }

                        }
                        else if(!lines_left.empty() && lines_right.empty()){
                                float ladian;
                                int degree;
                                it_left = lines_left.end() - 1;
                                ladian = atan2f((*it_left)[3]-(*it_left)[1], (*it_left)[2]-(*it_left)[0]);

                                degree = ladian * 180/CV_PI;
                                std::cout<<"            degree(left) : "<<degree<<std::endl;
                                if(abs(degree)>=20 && abs(degree)<=80){
                                        cv::line(hough_test,cv::Point(((*it_left)[0] + (*it_left)[2])/2,((*it_left)[1] + (*it_left)[3])/2),
                                        cv::Point(((*it_left)[0] + (*it_left)[2])/2,((*it_left)[1] + (*it_left)[3])/2+70),
                                        cv::Scalar(250,200,10),3,0);

                                        cv::line(hough_test,cv::Point((*it_left)[0], (*it_left)[1]),
                                        cv::Point((*it_left)[2],(*it_left)[3]),
                                        cv::Scalar(255,200,20),3,0);
      
                                }
                        }
                        else if(!lines_right.empty() && lines_left.empty()){
                                float ladian;
                                int degree;
                                it_right = lines_right.end() - 1;
                                ladian = atan2f((*it_right)[3]-(*it_right)[1], (*it_right)[2]-(*it_right)[0]);

                                degree = ladian * 180/CV_PI;
                                std::cout<<"            degree(right) : "<<degree<<std::endl;
                                if(abs(degree)>=0 && abs(degree)<=80){
                                        cv::line(hough_test,cv::Point(((*it_right)[0] + (*it_right)[2])/2,((*it_right)[1] + (*it_right)[3])/2),
                                        cv::Point(((*it_right)[0] + (*it_right)[2])/2,((*it_right)[1] + (*it_right)[3])/2+70),
                                        cv::Scalar(25,210,40),3,0);

                                        cv::line(hough_test,cv::Point((*it_right)[0], (*it_right)[1]),
                                        cv::Point((*it_right)[2],(*it_right)[3]),
                                        cv::Scalar(25,220,50),3,0);      
                                }
                        }
                              

                        cv::imshow("right hough",hough_test);

                        if (for_gui)
                        {
                                if (left_center_pt != -1 && right_center_pt != -1)
                                {
                                        l_r_center_pt = (left_center_pt + right_center_pt) / 2;
                                }
                                else if (left_center_pt != -1 && right_center_pt == -1)
                                {
                                        l_r_center_pt = left_center_pt;
                                }
                                else if (left_center_pt == -1 && right_center_pt != -1)
                                {
                                        l_r_center_pt = right_center_pt;
                                }
                                //cv::line(gui_test, cv::Point(img_center_left_pt, 100), cv::Point(img_center_left_pt, 300), cv::Scalar(100, 32, 100), 2);
                                //cv::line(gui_test, cv::Point(img_center_right_pt, 100), cv::Point(img_center_right_pt, 300), cv::Scalar(100, 32, 100), 2);
                                cv::line(gui_test, cv::Point(0, 165), cv::Point(gui_test.cols - 1, 165), cv::Scalar(23, 32, 100), 2);
                                cv::line(gui_test, cv::Point(0, 185), cv::Point(gui_test.cols - 1, 185), cv::Scalar(23, 32, 100), 2);
                        }

                        ////*detect traffic signal*////
                        if (!signal_lamp_checked && !blocking_bar_mode && !parking_mode && !signal_lamp_mode && !tunnel_mode)
                        {
                                // if((signalRedDetection(red2_hsv)) || (signalYellowDetection(yellow2_hsv))){
                                //         signal_lamp_mode = true;
                                // }
                                // if (signalRedDetection(red2_hsv))
                                // {
                                //         red_reliabilty++;
                                //         if (red_reliabilty > 7)
                                //         {
                                //                 signal_lamp_mode = true;
                                //         }
                                // }
                                // else if (signalYellowDetection(yellow2_hsv))
                                // {
                                //         yellow_reliabilty++;
                                //         if (yellow_reliabilty > 3)
                                //         {
                                //                 signal_lamp_mode = true;
                                //         }
                                // }
                               // if(signalGreenDetection(green_hsv)){
                                        signal_lamp_mode = true;
                               // }
                                if (signal_lamp_mode)
                                {
                                        normal_mode = false;
                                        std::cout << "################################ detect signal lamp ####################" << std::endl;
                                }
                        }

                        ////*detect blocking bar*////
                        if (!blocking_bar_checked && !blocking_bar_mode && !parking_mode && !signal_lamp_mode && !tunnel_mode)
                        {
                                blocking_bar_mode = detectBlockingBar(red_hsv);
                                if (blocking_bar_mode)
                                {
                                        normal_mode = false;
                                        std::cout << "################################ detect blocking bar ####################" << std::endl;
                                }
                        }

                        if (!parking_checked && !blocking_bar_mode && !parking_mode && !signal_lamp_mode && !tunnel_mode)
                        {
                                std::vector<cv::Point> dot_test_box_pt_w;
                                dot_test_box_pt_w.push_back(cv::Point(0, 0));
                                dot_test_box_pt_w.push_back(cv::Point(215, parking_white.rows - 1));

                                if (setMyLaneFitting(parking_white, dot_test_box_pt_w, "dot_test", dot_lane_fitting))
                                {

                                        if (dot_cnt != -1)
                                        {
                                                dot_cnt++;
                                                std::cout << "###################################################################### dot" << std::endl;
                                        }
                                        if (dot_cnt >= 2)
                                        {
                                                dot_cnt = -1;
                                                parking_mode = true;
                                                normal_mode = false;
                                        }
                                        cv::polylines(gui_test, dot_lane_fitting, 0, cv::Scalar(222, 232, 22), 3);
                                }
                        }
                        if (for_gui)
                        {
                                cv::imshow("gui_test", gui_test);
                        }
                        //left와 right인터벌을 뽑아내는 벡터 위치를 보정할지 말지 결정하자.

                        if (normal_mode)
                        {
                                x_goal_ = 0.07;
                                y_goal_ = 0.0;

                                std::cout << "************************************************************************" << std::endl;

                                if (left_interval != -1 && right_interval != -1)
                                {

                                        std::cout << "case 1 : 'all intervals are not -1'\n   pre left roi slope : " << pre_left_roi_slope << ",   left slope : " << left_roi_slope
                                                  << ",\n   pre right roi slope : " << pre_right_roi_slope << ",   right slope : " << right_roi_slope << std::endl;

                                        if ((left_interval > left_min_interval && left_interval < left_max_interval))
                                        { //go straight condition
                                                y_goal_ += 0.0;
                                        }
                                        else
                                        { //** left lane tracking condition

                                                if (left_interval <= left_min_interval)
                                                { //need right rotation(ang_vel<0 : right rotation)
                                                        if (abs(left_interval - left_min_interval) < 15)
                                                                y_goal_ = -0.18;
                                                        else if (abs(left_interval - left_min_interval) >= 15 && abs(left_interval - left_min_interval) < 30)
                                                                y_goal_ = -0.24;
                                                        else
                                                                y_goal_ = -0.28;
                                                }
                                                else
                                                { //need left rotation(ang_vel>0 : left rotation)

                                                        if (abs(left_interval - left_max_interval) < 15)
                                                                y_goal_ = 0.18;
                                                        else if (abs(left_interval - left_max_interval) >= 15 && abs(left_interval - left_max_interval) < 30)
                                                                y_goal_ = 0.24;
                                                        else
                                                                y_goal_ = 0.28;
                                                }
                                        }
                                        if (abs(left_interval - pre_left_interval) > 50 && left_interval != -1 && pre_left_interval != -1)
                                        {
                                                y_goal_ = prev_y_goal_;
                                        }
                                        if ((right_interval > right_min_interval && right_interval < right_max_interval))
                                        { //go straight condition
                                                y_goal_ += (float)0.0;
                                        }
                                        else
                                        {

                                                //** right lane tracking condition
                                                if (right_interval <= right_min_interval)
                                                { //need left rotation(ang_vel>0 : left rotation)
                                                        if (abs(right_interval - right_min_interval) < 15)
                                                                y_goal_ += (float)0.18;
                                                        else if (abs(right_interval - right_min_interval) >= 15 && abs(right_interval - right_min_interval) < 30)
                                                                y_goal_ += (float)0.24;
                                                        else
                                                                y_goal_ += (float)0.28;
                                                }
                                                else
                                                { //need right rotation(ang_vel<0 : right rotation)

                                                        if (abs(right_interval - right_max_interval) < 15)
                                                                y_goal_ += (float)-0.18;
                                                        else if (abs(right_interval - right_max_interval) >= 15 && abs(right_interval - right_max_interval) < 50)
                                                                y_goal_ += (float)-0.24;
                                                        else
                                                                y_goal_ += (float)-0.28;
                                                }
                                        }
                                        if (abs(right_interval - pre_right_interval) > 50 && right_interval != -1 && pre_right_interval != -1)
                                        {
                                                y_goal_ = prev_y_goal_;
                                        }

                                        ///***right doubling
                                        if ((abs(left_roi_slope) == 10 && abs(right_roi_slope) != 10) || left_roi_slope == 11)
                                        {
                                                if (right_roi_slope != 11)
                                                {
                                                        if (right_roi_slope <= 2)
                                                        {
                                                                if (right_roi_slope >= 0.7)
                                                                {
                                                                        y_goal_ = 0;
                                                                }
                                                                else if (right_roi_slope < 0.7 && right_roi_slope >= 0.6)
                                                                {
                                                                        y_goal_ = 0.21;
                                                                }
                                                                else if (right_roi_slope < 0.6 && right_roi_slope >= 0.5)
                                                                {
                                                                        y_goal_ = 0.24;
                                                                }
                                                                else if (right_roi_slope < 0.5 && right_roi_slope >= 0.4)
                                                                {
                                                                        y_goal_ = 0.28;
                                                                }
                                                                else if (right_roi_slope < 0.4 && right_roi_slope >= 0.3)
                                                                {
                                                                        y_goal_ = 0.29;
                                                                }
                                                                else
                                                                {
                                                                        y_goal_ = 0.31;
                                                                }
                                                        }
                                                }
                                                else
                                                {
                                                        ///***left line presents doubling to left but right line has no info***///
                                                        y_goal_ = -0.26;
                                                }
                                        }
                                        else if ((abs(left_roi_slope) != 10 && abs(right_roi_slope) == 10) || right_roi_slope == 11)
                                        {
                                                if (left_roi_slope != 11)
                                                {
                                                        if (abs(left_roi_slope) <= 2)
                                                        {
                                                                if (abs(left_roi_slope) >= 0.7)
                                                                {
                                                                        y_goal_ = 0;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.7 && abs(left_roi_slope) >= 0.6)
                                                                {
                                                                        y_goal_ = -0.21;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.6 && abs(left_roi_slope) >= 0.5)
                                                                {
                                                                        y_goal_ = -0.24;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.5 && abs(left_roi_slope) >= 0.4)
                                                                {
                                                                        y_goal_ = -0.28;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.4 && abs(left_roi_slope) >= 0.3)
                                                                {
                                                                        y_goal_ = -0.29;
                                                                }
                                                                else
                                                                {
                                                                        y_goal_ = -0.31;
                                                                }
                                                        }
                                                }
                                                else
                                                {
                                                        ///***right line presents doubling to right but left line has no info***///
                                                        y_goal_ = 0.26;
                                                }
                                        }
                                        else if (abs(left_roi_slope) == 11 && abs(right_roi_slope) == 11)
                                        {
                                                //  y_goal_ = prev_y_goal_;
                                        }
                                }
                                else if (left_interval != -1 && right_interval == -1)
                                {
                                        bool doubling_flag = false;
                                        std::cout << "case 2 : 'only left interval is not -1'\n pre left roi slope : " << pre_left_roi_slope << ", left slope : " << left_roi_slope
                                                  << ",\n pre right roi slope : " << pre_right_roi_slope << ", right slope : " << right_roi_slope << std::endl;

                                        if ((left_interval > left_min_interval && left_interval < left_max_interval))
                                        {
                                                if (abs(left_roi_slope) < 0.7)
                                                {
                                                        doubling_flag = true;
                                                }
                                                else
                                                {
                                                        //go straight condition
                                                        y_goal_ = 0;
                                                }
                                        }
                                        else
                                        {
                                                //** left lane tracking condition
                                                if (left_interval <= left_min_interval)
                                                { //need right rotation(ang_vel<0 : right rotation)
                                                        if (abs(left_interval - left_min_interval) <= 15)
                                                                y_goal_ = -0.19;
                                                        else if (abs(left_interval - left_min_interval) >= 15 && abs(left_interval - left_min_interval) < 30)
                                                                y_goal_ = -0.24;
                                                        else
                                                                y_goal_ = -0.28;
                                                }
                                                else
                                                { //need left rotation(ang_vel>0 : left rotation)
                                                        if (abs(left_interval - left_max_interval) <= 15)
                                                                y_goal_ = 0.15;
                                                        else if (abs(left_interval - left_max_interval) >= 15 && abs(left_interval - left_max_interval) < 30)
                                                                y_goal_ = 0.21;
                                                        else
                                                                y_goal_ = 0.26;
                                                }
                                        }
                                        if (abs(left_interval - pre_left_interval) > 40 && left_interval != -1 && pre_left_interval != -1)
                                        {
                                                y_goal_ = prev_y_goal_;
                                        }
                                        if (((abs(left_roi_slope) != 10 && abs(right_roi_slope) == 10)) || doubling_flag)
                                        {
                                                if (left_roi_slope != 11)
                                                {
                                                        if (abs(left_roi_slope) <= 2)
                                                        {
                                                                if (abs(left_roi_slope) >= 0.75)
                                                                {
                                                                        y_goal_ = 0;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.75 && abs(left_roi_slope) >= 0.6)
                                                                {
                                                                        y_goal_ = -0.19;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.6 && abs(left_roi_slope) >= 0.5)
                                                                {
                                                                        y_goal_ = -0.24;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.5 && abs(left_roi_slope) >= 0.4)
                                                                {
                                                                        y_goal_ = -0.28;
                                                                }
                                                                else if (abs(left_roi_slope) < 0.4 && abs(left_roi_slope) >= 0.3)
                                                                {
                                                                        y_goal_ = -0.31;
                                                                }
                                                                else
                                                                {
                                                                        y_goal_ = -0.33;
                                                                }
                                                        }
                                                }
                                                else
                                                {
                                                        ///***right line presents doubling to right but left line has no info***///
                                                        y_goal_ = 0.26;
                                                }
                                        }
                                        else if (abs(left_roi_slope) == 11 && abs(right_roi_slope) == 11)
                                        {
                                                y_goal_ = prev_y_goal_;
                                        }
                                }
                                else if (left_interval == -1 && right_interval != -1)
                                {
                                        bool doubling_flag = false;
                                        std::cout << "case 3 : 'only right interval is not -1'\n pre left roi slope : " << pre_left_roi_slope << ", left slope : " << left_roi_slope
                                                  << ",\n pre right roi slope : " << pre_right_roi_slope << ", right slope : " << right_roi_slope << std::endl;
                                        if (right_roi_slope < 0.7)
                                        {
                                                doubling_flag = true;
                                        }
                                        if ((right_interval > right_min_interval && right_interval < right_max_interval))
                                        {
                                                if (right_roi_slope < 0.8)
                                                {
                                                        doubling_flag = true;
                                                }
                                                else
                                                {
                                                        //go straight condition
                                                        y_goal_ = 0;
                                                }
                                        }

                                        else
                                        {

                                                //** right lane tracking condition
                                                if (right_interval <= right_min_interval)
                                                { //need left rotation(ang_vel>0 : left rotation)
                                                        if (abs(right_interval - right_min_interval) <= 15)
                                                                y_goal_ = 0.19;
                                                        else if (abs(right_interval - right_min_interval) > 15 && abs(right_interval - right_min_interval) <= 30)
                                                                y_goal_ = 0.24;
                                                        else
                                                                y_goal_ = 0.26;
                                                }
                                                else
                                                { //need right rotation(ang_vel<0 : right rotation)
                                                        if (abs(right_interval - right_max_interval) <= 15)
                                                                y_goal_ = -0.19;
                                                        else if (abs(right_interval - right_max_interval) > 15 && abs(right_interval - right_max_interval) <= 30)
                                                                y_goal_ = -0.24;
                                                        else
                                                                y_goal_ = -0.26;
                                                }
                                        }
                                        if (abs(right_interval - pre_right_interval) > 50 && right_interval != -1 && pre_right_interval != -1)
                                        {
                                                y_goal_ = prev_y_goal_;
                                        }
                                        if (((abs(left_roi_slope) == 10 && abs(right_roi_slope) != 10)) || doubling_flag)
                                        {
                                                if (right_roi_slope != 11)
                                                {
                                                        if (right_roi_slope <= 2)
                                                        {
                                                                if (right_roi_slope >= 0.8)
                                                                {
                                                                        y_goal_ = 0;
                                                                }
                                                                else if (right_roi_slope < 0.8 && right_roi_slope >= 0.7)
                                                                {
                                                                        y_goal_ = 0.21;
                                                                }
                                                                else if (right_roi_slope < 0.7 && right_roi_slope >= 0.6)
                                                                {

                                                                        y_goal_ = 0.27;
                                                                }
                                                                else if (right_roi_slope < 0.6 && right_roi_slope >= 0.5)
                                                                {
                                                                        y_goal_ = 0.29;
                                                                }
                                                                else if (right_roi_slope < 0.5 && right_roi_slope >= 0.4)
                                                                {
                                                                        y_goal_ = 0.30;
                                                                }
                                                                else
                                                                {
                                                                        y_goal_ = 0.32;
                                                                }
                                                        }
                                                }
                                                else
                                                {
                                                        ///***left line presents doubling to left but right line has no info***///
                                                        y_goal_ = -0.26;
                                                }
                                        }
                                        else if (abs(left_roi_slope) == 11 && abs(right_roi_slope) == 11)
                                        {
                                                //y_goal_ = prev_y_goal_;
                                        }
                                }
                                else
                                {
                                        std::cout << "case 4: 'all intervals are -1'\n pre left roi slope : " << pre_left_roi_slope << ", left slope : " << left_roi_slope
                                                  << ",\n pre right roi slope : " << pre_right_roi_slope << ", right slope : " << right_roi_slope << std::endl;

                                        y_goal_ = prev_y_goal_;
                                }

                                // if (blocking_bar_checked)
                                // {
                                //         y_goal_ = 0;
                                //         if (right_interval < right_min_interval)
                                //                 y_goal_ = -0.1;
                                //         else if (right_interval > right_max_interval)
                                //                 y_goal_ = 0.1;
                                // }
                                if (abs(y_goal_) > 0.25)
                                {
                                        // x_goal_ = 0.06;
                                }
                                std::cout << "x_goal_            : " << x_goal_ << std::endl;
                                std::cout << "prev_y_goal        : " << prev_y_goal_ << ",  y_goal_         : " << y_goal_ << std::endl;
                                std::cout << "pre_right_size     : " << pre_right_size << ",     right_lane_size : " << right_lane_fitting.size() << std::endl;
                                std::cout << "pre_left_size      : " << pre_left_size << ",     left_lane_size  : " << left_lane_fitting.size() << std::endl;
                                std::cout << "pre_right_interval : " << pre_right_interval << ",    right_interval  : " << right_interval << std::endl;
                                std::cout << "pre_left_interval  : " << pre_left_interval << ",      left_interval  : " << left_interval << std::endl;
                                std::cout << "************************************************************************" << std::endl;
                                goal_array.data.clear();
                                goal_array.data.resize(0);
                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                goal_array.data.push_back(y_goal_);
                                //////////////////////////////////////////nomal아닐때도 갱신 되도록....?
                                prev_y_goal_ = y_goal_;
                                msg_count_left = 0;
                                msg_count_right = 0;
                                left_interval_vec.clear();
                                right_interval_vec.clear();
                                left_interval_vec.resize(0);
                                right_interval_vec.resize(0);
                                pre_left_interval = left_interval;
                                pre_right_interval = right_interval;
                                pre_left_size = left_lane_fitting.size();
                                pre_right_size = right_lane_fitting.size();
                                left_roi_slope = 11;
                                right_roi_slope = 11;
                                pre_left_roi_slope = left_roi_slope;
                                pre_right_roi_slope = right_roi_slope;
                                left_interval = -1;
                                right_interval = -1;
                        }

                        //  cv::Mat ss = frame.clone();
                        if (parking_mode && !blocking_bar_mode && !signal_lamp_mode && !tunnel_mode)
                        {

                                cv::Mat parked = origin_white_hsv.clone();

                                cv::Mat park_zero = cv::Mat::zeros(origin_white_hsv.size(), CV_8UC1);
                                std::cout << "parking stage : " << parking_stage << std::endl;
                                switch (parking_stage)
                                {
                                case 0:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        // if ((left_roi_slope < 0|| left_roi_slope == 11) && left_lane_fitting.size() < 20)
                                        // {
                                        //         std::cout<<"111111111"<<std::endl;
                                        //         goal_array.data.clear();
                                        //         goal_array.data.resize(0);
                                        //         x_goal_ = 0;
                                        //         y_goal_ = 0.1;
                                        //         goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        //         goal_array.data.push_back(y_goal_);
                                        // }
                                        // else if (left_roi_slope < 0 && left_lane_fitting.size() >= 20)
                                        // {
                                        //     std::cout<<"2222222"<<std::endl;
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        parking_stage = 1;
                                        go_cnt = 0;
                                        // }

                                        break;
                                case 1:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0.1;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 6)
                                        {
                                                parking_stage = 2;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 2:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {
                                                parking_stage = 3;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 3:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = -0.45;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 12)
                                        {
                                                parking_stage = 4;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 4:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 2)
                                        {
                                                parking_stage = 5;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 5:
                                {

                                        cv::Mat park_zero = cv::Mat::zeros(park_detect.size(), CV_8UC3);
                                        std::vector<cv::Point> park_line_test;
                                        for (int x = 80; x < 175; x++)
                                        {
                                                for (int y = 160; y > 70; y--)
                                                {
                                                        uchar *park_pt = park_detect.ptr<uchar>(y);
                                                        uchar *parkline_data = park_zero.ptr<uchar>(y);
                                                        if (park_pt[x] != (uchar)0)
                                                        {
                                                                park_line_test.push_back(cv::Point(x, y));
                                                                parkline_data[x] = (uchar)255;
                                                                break;
                                                        }
                                                }
                                        }
                                        int line_max = -1, line_tmp = 0;
                                        for (int y = 140; y > 100; y--)
                                        {
                                                for (int i = 0; i < park_line_test.size(); i++)
                                                {
                                                        if (park_line_test[i].y == y)
                                                        {
                                                                for (int x = park_detect.cols / 2 - 30; x < park_detect.cols / 2 + 30; x++)
                                                                {
                                                                        if (park_line_test[i].x == x)
                                                                        {
                                                                                line_tmp++;
                                                                        }
                                                                }
                                                        }
                                                }
                                                if (line_tmp > line_max)
                                                {
                                                        line_max = line_tmp;
                                                        line_tmp = 0;
                                                }
                                        }
                                        std::cout << "park line_tmp : " << line_tmp << std::endl;

                                        go_cnt++;
                                        if (line_max > 20)
                                        {
                                                parking_reliabilty++;
                                        }
                                        if (go_cnt > 7)
                                        {
                                                if (parking_reliabilty > 4)
                                                {
                                                        parking_stage = 6;
                                                        go_cnt = 0;
                                                }
                                                else
                                                {
                                                        parking_stage = 13;
                                                        go_cnt = 0;
                                                }
                                        }
                                }
                                break;
                                case 6: //first parking area
                                {
                                        if (go_cnt < 16)
                                        {
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.1;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                        }
                                        else
                                        {
                                                go_cnt = 0;
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);

                                                parking_stage = 7;
                                        }
                                }
                                break;
                                case 7: //first parking area
                                {
                                        if (go_cnt < 1)
                                        {
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                        }
                                        else
                                        {
                                                go_cnt = 0;
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);

                                                parking_stage = 8;
                                        }
                                }
                                break;
                                case 8:
                                        if (go_cnt < 17)
                                        {
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = -0.1;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                        }
                                        else
                                        {
                                                go_cnt = 0;
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                parking_stage = 9;
                                        }
                                        break;
                                case 9:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 2)
                                        {

                                                parking_stage = 10;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 10:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0.45;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 12)
                                        {

                                                parking_stage = 11;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 11:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 2)
                                        {
                                                parking_stage = 12;
                                                go_cnt = 0;
                                        }

                                        break;
                                case 12:
                                        normal_mode = true;
                                        parking_mode = false;
                                        parking_checked = true;
                                        break; ///////////
                                case 13:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {
                                                parking_stage = 14;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 14:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0.45;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 16)
                                        {

                                                parking_stage = 15;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 15:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {
                                                parking_stage = 16;
                                                go_cnt = 0;
                                                //
                                        }
                                        break;
                                case 16:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0.1;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 20)
                                        {
                                                parking_stage = 17;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 17:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 2)
                                        {
                                                parking_stage = 18;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 18:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = -0.45;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 16)
                                        {
                                                parking_stage = 19;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 19:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {
                                                parking_stage = 20;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 20:
                                {
                                        if (go_cnt < 13)
                                        {
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0.1;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                        }
                                        else
                                        {
                                                go_cnt = 0;
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
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
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 3)
                                        {
                                                parking_stage = 22;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 22:
                                        if (go_cnt < 13)
                                        {
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = -0.1;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                go_cnt++;
                                        }
                                        else
                                        {
                                                go_cnt = 0;
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                                goal_array.data.push_back(y_goal_);
                                                parking_stage = 23;
                                        }
                                        break;
                                case 23:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {

                                                parking_stage = 24;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 24:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0.45;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 16)
                                        {

                                                parking_stage = 25;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 25:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 5)
                                        {
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
                        }

                        if (blocking_bar_mode && !parking_mode && !signal_lamp_mode && !tunnel_mode)
                        {
                                switch (blocking_bar_stage)
                                {
                                case 0:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0.0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 3)
                                        {
                                                blocking_bar_stage = 1;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 1:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 10 && !detectBlockingBar(red_hsv))
                                        {
                                                blocking_bar_stage = 3;
                                                go_cnt = 0;
                                        }
                                        else if (go_cnt > 10 && detectBlockingBar(red_hsv))
                                        {
                                                blocking_bar_stage = 5;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 2:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_);
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {
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
                                        if (go_cnt > 2)
                                        {
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
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_);
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 1)
                                        {
                                                blocking_bar_stage = 6;
                                                go_cnt = 0;
                                        }
                                        break;
                                case 6:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_); //linear_vel아님 정확힌 직진방향 x목표점임. 속도변수따로만들기
                                        goal_array.data.push_back(y_goal_);
                                        go_cnt++;
                                        if (go_cnt > 30 && !detectBlockingBar(red_hsv))
                                        {
                                                blocking_bar_stage = 2;
                                                go_cnt = 0;
                                        }
                                        else if (go_cnt > 30 && detectBlockingBar(red_hsv))
                                        {
                                                go_cnt = 0;
                                                blocking_bar_mode = 0;
                                                blocking_bar_checked = false;
                                                blocking_bar_stage = 0;
                                        }
                                        break;
                                }
                        }

                        if (signal_lamp_mode && !parking_mode && !blocking_bar_mode && !tunnel_mode)
                        {
                                switch (signal_lamp_stage)
                                {
                                case 0:
                                        goal_array.data.clear();
                                        goal_array.data.resize(0);
                                        x_goal_ = 0;
                                        y_goal_ = 0;
                                        goal_array.data.push_back(x_goal_);
                                        goal_array.data.push_back(y_goal_);
                                        //go_cnt++;
                                        signal_lamp_stage = 0;
                                        if(signalGreenDetection(green_hsv)){
                                                std::cout<<"green detection!~~~~~~~~~~~~~~~~~~~~~~~~\n";
                                                goal_array.data.clear();
                                                goal_array.data.resize(0);
                                                x_goal_ = 0;
                                                y_goal_ = 0;
                                                goal_array.data.push_back(x_goal_);
                                                goal_array.data.push_back(y_goal_);
                                                normal_mode = true;
                                                signal_lamp_mode = false;
                                                signal_lamp_checked = true;
                                                go_cnt = 0;
                                        }
                                        // //if (go_cnt > 3)
                                        // //{
                                        // signal_lamp_stage = 1;
                                        // //        go_cnt = 0;
                                        // //}
                                        // if (red_rotation > 10)
                                        // {
                                        //         goal_array.data.clear();
                                        //         goal_array.data.resize(0);
                                        //         x_goal_ = 0;
                                        //         y_goal_ = 0;
                                        //         goal_array.data.push_back(x_goal_);
                                        //         goal_array.data.push_back(y_goal_);
                                        //         normal_mode = true;
                                        //         signal_lamp_mode = false;
                                        //         signal_lamp_checked = false;
                                        //         red_rotation = 0;
                                        //         go_cnt = 0;
                                        // }
                                        break;
                                // case 1:
                                //         if (signalGreenDetection(green_hsv))
                                //         {
                                //                 red_reliabilty = 0;
                                //                 yellow_reliabilty = 0;
                                //                 red_rotation = 0;
                                //                 yellow_rotation = 0;
                                //                 goal_array.data.clear();
                                //                 goal_array.data.resize(0);
                                //                 x_goal_ = 0;
                                //                 y_goal_ = 0;
                                //                 goal_array.data.push_back(x_goal_);
                                //                 goal_array.data.push_back(y_goal_);
                                //                 normal_mode = true;
                                //                 signal_lamp_mode = false;
                                //                 signal_lamp_checked = true;
                                //                 go_cnt = 0;
                                //         }
                                //         else if (signalRedDetection(red2_hsv))
                                //         {
                                //                 goal_array.data.clear();
                                //                 goal_array.data.resize(0);
                                //                 x_goal_ = 0;
                                //                 y_goal_ = 0;
                                //                 goal_array.data.push_back(x_goal_);
                                //                 goal_array.data.push_back(y_goal_);
                                //                 go_cnt = 0;
                                //                 red_rotation++;
                                //                 if (red_rotation > 15)
                                //                 {
                                //                         goal_array.data.clear();
                                //                         goal_array.data.resize(0);
                                //                         x_goal_ = 0.8;
                                //                         y_goal_ = 0;
                                //                         goal_array.data.push_back(x_goal_);
                                //                         goal_array.data.push_back(y_goal_);
                                //                         normal_mode = true;
                                //                         signal_lamp_mode = false;
                                //                         signal_lamp_checked = false;
                                //                         red_rotation = 0;
                                //                         go_cnt = 0;
                                //                 }
                                //         }
                                //         else if (signalYellowDetection(yellow2_hsv))
                                //         {
                                //                 goal_array.data.clear();
                                //                 goal_array.data.resize(0);
                                //                 x_goal_ = 0;
                                //                 y_goal_ = 0;
                                //                 goal_array.data.push_back(x_goal_);
                                //                 goal_array.data.push_back(y_goal_);
                                //                 go_cnt = 0;
                                //                 yellow_rotation++;
                                //                 if (yellow_rotation > 5)
                                //                 {
                                //                         goal_array.data.clear();
                                //                         goal_array.data.resize(0);
                                //                         x_goal_ = 0.08;
                                //                         y_goal_ = 0;
                                //                         goal_array.data.push_back(x_goal_);
                                //                         goal_array.data.push_back(y_goal_);
                                //                         normal_mode = true;
                                //                         signal_lamp_mode = false;
                                //                         signal_lamp_checked = false;
                                //                         yellow_rotation = 0;
                                //                         go_cnt = 0;
                                //                 }
                                //         }

                                //         break;
                                // case 2:
                                //         goal_array.data.clear();
                                //         goal_array.data.resize(0);
                                //         x_goal_ = 0;
                                //         y_goal_ = 0;
                                //         goal_array.data.push_back(x_goal_);
                                //         goal_array.data.push_back(y_goal_);
                                //         normal_mode = true;
                                //         signal_lamp_mode = false;
                                //         signal_lamp_checked = true;
                                //         go_cnt = 0;
                                //         break;
                                // case 3:
                                //         goal_array.data.clear();
                                //         goal_array.data.resize(0);
                                //         x_goal_ = 0;
                                //         y_goal_ = 0;
                                //         goal_array.data.push_back(x_goal_);
                                //         goal_array.data.push_back(y_goal_);
                                //         normal_mode = true;
                                //         signal_lamp_mode = false;
                                //         signal_lamp_checked = false;
                                //         red_rotation = 0;
                                //         yellow_rotation = 0;
                                //         go_cnt = 0;
                                //         break;
                                }
                        }

                        ////*Restore birdeyeview img to origin view*////
                        restoreImgWithLangeMerge(callane, frame, yellow_labeling, white_labeling, mergelane);

                        ////*Make lane infomation msg for translate scan data*////
                        extractLanePoint(gui_test, mergelane);

                        output_origin_for_copy = origin.clone();
                }
                else
                { //frame is empty
                        while (frame.empty())
                        { //for unplugged camera
                                cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
                                frame = cv_ptr->image;
                        }
                        x_goal_ = 0;
                        y_goal_ = 0.; //제자리회전시켜보기
                }
        }
        catch (cv_bridge::Exception &e)
        {
                ROS_ERROR("cv_bridge exception : %s", e.what());
                return;
        }

        if (auto_shot)
        {
                std::cout << "Save screen shot" << std::endl;
                cv::imwrite("/home/seuleee/Desktop/autorace_img_src/1110/right_sign_" + to_string(imgNum) + ".jpg", output_origin_for_copy);
                imgNum++;
        }
        int ckey = cv::waitKey(10);
        if (ckey == 27)
                exit(1);
        else if (ckey == 32)
        { //For save using space key
                std::cout << "Save screen shot" << std::endl;
                cv::imwrite("/home/kim/catkin_ws/src/sign_signa_" + to_string(imgNum) + ".jpg", output_origin_for_copy);
                imgNum++;
        }
}

void InitImgObjectforROS::initParam()
{
        nh.param<int>("/" + groupName + "/lane_detection/debug", debug, 0);
        nh.param<int>("/" + groupName + "/lane_detection/web_cam", web_cam, 0);
        nh.param<int>("/" + groupName + "/lane_detection/imshow", imshow, 0);
        nh.param<int>("/" + groupName + "/lane_detection/track_bar", track_bar, 0);

        nh.param<int>("/" + groupName + "/lane_detection/for_gui", for_gui, 0);
        nh.param<int>("/" + groupName + "/lane_detection/gazebo", gazebo, 1);
        nh.param<int>("/" + groupName + "/lane_detection/bird_eye_view", bird_eye_view, 0);
        nh.param<int>("/" + groupName + "/lane_detection/auto_shot", auto_shot, 0);
        nh.param<int>("/" + groupName + "/lane_detection/auto_record", auto_record, 0);

        nh.param<int>("/" + groupName + "/lane_detection/left_min_interval", left_min_interval, 110);
        nh.param<int>("/" + groupName + "/lane_detection/left_max_interval", left_max_interval, 150);
        nh.param<int>("/" + groupName + "/lane_detection/right_min_interval", right_min_interval, 120);
        nh.param<int>("/" + groupName + "/lane_detection/right_max_interval", right_max_interval, 160);

        nh.param<int>("/" + groupName + "/lane_detection/reset_msg", reset_msg, 0);
        //default yellow when no launch file for lane color
        nh.param<int>("/" + groupName + "/lane_detection/y_hmin", y_hmin, 15);
        nh.param<int>("/" + groupName + "/lane_detection/y_hmax", y_hmax, 21);
        nh.param<int>("/" + groupName + "/lane_detection/y_smin", y_smin, 52);
        nh.param<int>("/" + groupName + "/lane_detection/y_smax", y_smax, 151);
        nh.param<int>("/" + groupName + "/lane_detection/y_vmin", y_vmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/y_vmax", y_vmax, 180);
        //default white when no launch file for lane color
        nh.param<int>("/" + groupName + "/lane_detection/w_hmin", w_hmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/w_hmax", w_hmax, 180);
        nh.param<int>("/" + groupName + "/lane_detection/w_smin", w_smin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/w_smax", w_smax, 24);
        nh.param<int>("/" + groupName + "/lane_detection/w_vmin", w_vmin, 172);
        nh.param<int>("/" + groupName + "/lane_detection/w_vmax", w_vmax, 255);

        //default red when no launch file for blocking bar
        nh.param<int>("/" + groupName + "/lane_detection/r_hmin", r_hmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/r_hmax", r_hmax, 180);
        nh.param<int>("/" + groupName + "/lane_detection/r_smin", r_smin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/r_smax", r_smax, 24);
        nh.param<int>("/" + groupName + "/lane_detection/r_vmin", r_vmin, 172);
        nh.param<int>("/" + groupName + "/lane_detection/r_vmax", r_vmax, 255);

        //default red when no launch file for signal lamp
        nh.param<int>("/" + groupName + "/lane_detection/r2_hmin", r2_hmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/r2_hmax", r2_hmax, 180);
        nh.param<int>("/" + groupName + "/lane_detection/r2_smin", r2_smin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/r2_smax", r2_smax, 24);
        nh.param<int>("/" + groupName + "/lane_detection/r2_vmin", r2_vmin, 172);
        nh.param<int>("/" + groupName + "/lane_detection/r2_vmax", r2_vmax, 255);
        //default yellow when no launch file for signal lamp
        nh.param<int>("/" + groupName + "/lane_detection/y2_hmin", y2_hmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/y2_hmax", y2_hmax, 180);
        nh.param<int>("/" + groupName + "/lane_detection/y2_smin", y2_smin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/y2_smax", y2_smax, 24);
        nh.param<int>("/" + groupName + "/lane_detection/y2_vmin", y2_vmin, 172);
        nh.param<int>("/" + groupName + "/lane_detection/y2_vmax", y2_vmax, 255);
        //default green when no launch file for signal lamp
        nh.param<int>("/" + groupName + "/lane_detection/g_hmin", g_hmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/g_hmax", g_hmax, 180);
        nh.param<int>("/" + groupName + "/lane_detection/g_smin", g_smin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/g_smax", g_smax, 24);
        nh.param<int>("/" + groupName + "/lane_detection/g_vmin", g_vmin, 172);
        nh.param<int>("/" + groupName + "/lane_detection/g_vmax", g_vmax, 255);

        //default blue when no launch file for street sign
        nh.param<int>("/" + groupName + "/lane_detection/b_hmin", b_hmin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/b_hmax", b_hmax, 180);
        nh.param<int>("/" + groupName + "/lane_detection/b_smin", b_smin, 0);
        nh.param<int>("/" + groupName + "/lane_detection/b_smax", b_smax, 24);
        nh.param<int>("/" + groupName + "/lane_detection/b_vmin", b_vmin, 172);
        nh.param<int>("/" + groupName + "/lane_detection/b_vmax", b_vmax, 255);
        ROS_INFO("vision gazebo %d\n", gazebo);
}

void InitImgObjectforROS::initMyRESETTrackbar(const string &trackbar_name)
{
        cv::namedWindow(trackbar_name, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("reset msg", trackbar_name, &reset_msg, 1, NULL);
        cv::setTrackbarPos("reset msg", trackbar_name, reset_msg);
}
void InitImgObjectforROS::setMyRESETTrackbarValue(const string &trackbar_name)
{
        reset_msg = cv::getTrackbarPos("reset msg", trackbar_name);
        nh.setParam("/" + groupName + "/lane_detection/reset_msg", reset_msg);
}

void InitImgObjectforROS::initMyHSVTrackbar(const string &trackbar_name)
{

        cv::namedWindow(trackbar_name, cv::WINDOW_AUTOSIZE);
        if (trackbar_name.find("YELLOW") != string::npos && trackbar_name.find("LANE") != string::npos)
        {

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
        else if (trackbar_name.find("WHITE") != string::npos && trackbar_name.find("LANE") != string::npos)
        {

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
        else if (trackbar_name.find("BLOCKING") != string::npos && trackbar_name.find("RED") != string::npos)
        {

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
        else if (trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("RED") != string::npos)
        {

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
        else if (trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("YELLOW") != string::npos)
        {

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
        else if (trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("GREEN") != string::npos)
        {

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
        else if (trackbar_name.find("BLUE") != string::npos && trackbar_name.find("AREA") != string::npos)
        {

                cv::createTrackbar("h min", trackbar_name, &b_hmin, 179, NULL);
                cv::setTrackbarPos("h min", trackbar_name, b_hmin);

                cv::createTrackbar("h max", trackbar_name, &b_hmax, 179, NULL);
                cv::setTrackbarPos("h max", trackbar_name, b_hmax);

                cv::createTrackbar("s min", trackbar_name, &b_smin, 255, NULL);
                cv::setTrackbarPos("s min", trackbar_name, b_smin);

                cv::createTrackbar("s max", trackbar_name, &b_smax, 255, NULL);
                cv::setTrackbarPos("s max", trackbar_name, b_smax);

                cv::createTrackbar("v min", trackbar_name, &b_vmin, 255, NULL);
                cv::setTrackbarPos("v min", trackbar_name, b_vmin);

                cv::createTrackbar("v max", trackbar_name, &b_vmax, 255, NULL);
                cv::setTrackbarPos("v max", trackbar_name, b_vmax);
        }
}

void InitImgObjectforROS::setMyHSVTrackbarValue(const string &trackbar_name)
{

        if (trackbar_name.find("YELLOW") != string::npos && trackbar_name.find("LANE") != string::npos)
        {
                y_hmin = cv::getTrackbarPos("h min", trackbar_name);
                y_hmax = cv::getTrackbarPos("h max", trackbar_name);
                y_smin = cv::getTrackbarPos("s min", trackbar_name);
                y_smax = cv::getTrackbarPos("s max", trackbar_name);
                y_vmin = cv::getTrackbarPos("v min", trackbar_name);
                y_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }
        else if (trackbar_name.find("WHITE") != string::npos && trackbar_name.find("LANE") != string::npos)
        {
                w_hmin = cv::getTrackbarPos("h min", trackbar_name);
                w_hmax = cv::getTrackbarPos("h max", trackbar_name);
                w_smin = cv::getTrackbarPos("s min", trackbar_name);
                w_smax = cv::getTrackbarPos("s max", trackbar_name);
                w_vmin = cv::getTrackbarPos("v min", trackbar_name);
                w_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }
        else if (trackbar_name.find("BLOCKING_RED") != string::npos)
        {

                r_hmin = cv::getTrackbarPos("h min", trackbar_name);
                r_hmax = cv::getTrackbarPos("h max", trackbar_name);
                r_smin = cv::getTrackbarPos("s min", trackbar_name);
                r_smax = cv::getTrackbarPos("s max", trackbar_name);
                r_vmin = cv::getTrackbarPos("v min", trackbar_name);
                r_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }
        else if (trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("RED") != string::npos)
        {

                r2_hmin = cv::getTrackbarPos("h min", trackbar_name);
                r2_hmax = cv::getTrackbarPos("h max", trackbar_name);
                r2_smin = cv::getTrackbarPos("s min", trackbar_name);
                r2_smax = cv::getTrackbarPos("s max", trackbar_name);
                r2_vmin = cv::getTrackbarPos("v min", trackbar_name);
                r2_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }
        else if (trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("RED") != string::npos)
        {

                y2_hmin = cv::getTrackbarPos("h min", trackbar_name);
                y2_hmax = cv::getTrackbarPos("h max", trackbar_name);
                y2_smin = cv::getTrackbarPos("s min", trackbar_name);
                y2_smax = cv::getTrackbarPos("s max", trackbar_name);
                y2_vmin = cv::getTrackbarPos("v min", trackbar_name);
                y2_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }
        else if (trackbar_name.find("SIGNAL") != string::npos && trackbar_name.find("GREEN") != string::npos)
        {

                g_hmin = cv::getTrackbarPos("h min", trackbar_name);
                g_hmax = cv::getTrackbarPos("h max", trackbar_name);
                g_smin = cv::getTrackbarPos("s min", trackbar_name);
                g_smax = cv::getTrackbarPos("s max", trackbar_name);
                g_vmin = cv::getTrackbarPos("v min", trackbar_name);
                g_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }
        else if (trackbar_name.find("BLUE") != string::npos && trackbar_name.find("AREA") != string::npos)
        {

                b_hmin = cv::getTrackbarPos("h min", trackbar_name);
                b_hmax = cv::getTrackbarPos("h max", trackbar_name);
                b_smin = cv::getTrackbarPos("s min", trackbar_name);
                b_smax = cv::getTrackbarPos("s max", trackbar_name);
                b_vmin = cv::getTrackbarPos("v min", trackbar_name);
                b_vmax = cv::getTrackbarPos("v max", trackbar_name);
        }

        nh.setParam("/" + groupName + "/lane_detection/y_hmin", y_hmin);
        nh.setParam("/" + groupName + "/lane_detection/y_hmax", y_hmax);
        nh.setParam("/" + groupName + "/lane_detection/y_smin", y_smin);
        nh.setParam("/" + groupName + "/lane_detection/y_smax", y_smax);
        nh.setParam("/" + groupName + "/lane_detection/y_vmin", y_vmin);
        nh.setParam("/" + groupName + "/lane_detection/y_vmax", y_vmax);

        nh.setParam("/" + groupName + "/lane_detection/w_hmin", w_hmin);
        nh.setParam("/" + groupName + "/lane_detection/w_hmax", w_hmax);
        nh.setParam("/" + groupName + "/lane_detection/w_smin", w_smin);
        nh.setParam("/" + groupName + "/lane_detection/w_smax", w_smax);
        nh.setParam("/" + groupName + "/lane_detection/w_vmin", w_vmin);
        nh.setParam("/" + groupName + "/lane_detection/w_vmax", w_vmax);

        nh.setParam("/" + groupName + "/lane_detection/r_hmin", r_hmin);
        nh.setParam("/" + groupName + "/lane_detection/r_hmax", r_hmax);
        nh.setParam("/" + groupName + "/lane_detection/r_smin", r_smin);
        nh.setParam("/" + groupName + "/lane_detection/r_smax", r_smax);
        nh.setParam("/" + groupName + "/lane_detection/r_vmin", r_vmin);
        nh.setParam("/" + groupName + "/lane_detection/r_vmax", r_vmax);

        nh.setParam("/" + groupName + "/lane_detection/r2_hmin", r2_hmin);
        nh.setParam("/" + groupName + "/lane_detection/r2_hmax", r2_hmax);
        nh.setParam("/" + groupName + "/lane_detection/r2_smin", r2_smin);
        nh.setParam("/" + groupName + "/lane_detection/r2_smax", r2_smax);
        nh.setParam("/" + groupName + "/lane_detection/r2_vmin", r2_vmin);
        nh.setParam("/" + groupName + "/lane_detection/r2_vmax", r2_vmax);

        nh.setParam("/" + groupName + "/lane_detection/y2_hmin", y2_hmin);
        nh.setParam("/" + groupName + "/lane_detection/y2_hmax", y2_hmax);
        nh.setParam("/" + groupName + "/lane_detection/y2_smin", y2_smin);
        nh.setParam("/" + groupName + "/lane_detection/y2_smax", y2_smax);
        nh.setParam("/" + groupName + "/lane_detection/y2_vmin", y2_vmin);
        nh.setParam("/" + groupName + "/lane_detection/y2_vmax", y2_vmax);

        nh.setParam("/" + groupName + "/lane_detection/g_hmin", g_hmin);
        nh.setParam("/" + groupName + "/lane_detection/g_hmax", g_hmax);
        nh.setParam("/" + groupName + "/lane_detection/g_smin", g_smin);
        nh.setParam("/" + groupName + "/lane_detection/g_smax", g_smax);
        nh.setParam("/" + groupName + "/lane_detection/g_vmin", g_vmin);
        nh.setParam("/" + groupName + "/lane_detection/g_vmax", g_vmax);

        nh.setParam("/" + groupName + "/lane_detection/b_hmin", b_hmin);
        nh.setParam("/" + groupName + "/lane_detection/b_hmax", b_hmax);
        nh.setParam("/" + groupName + "/lane_detection/b_smin", b_smin);
        nh.setParam("/" + groupName + "/lane_detection/b_smax", b_smax);
        nh.setParam("/" + groupName + "/lane_detection/b_vmin", b_vmin);
        nh.setParam("/" + groupName + "/lane_detection/b_vmax", b_vmax);
}

void InitImgObjectforROS::setColorPreocessing(lane_detect_algo::CalLane callane, cv::Mat src, cv::Mat &dst_y, cv::Mat &dst_w, cv::Mat &dst_r, 
                                              cv::Mat &dst_r2, cv::Mat &dst_g, cv::Mat &dst_y2, cv::Mat &dst_b, cv::Mat &parking_white)
{
        ////*Make trackbar obj*////
        if (track_bar)
        {
                setMyHSVTrackbarValue(groupName + "_YELLOW_LANE_TRACKBAR");
                setMyHSVTrackbarValue(groupName + "_WHITE_LANE_TRACKBAR");
                setMyHSVTrackbarValue(groupName + "_BLOCKING_RED_TRACKBAR");
                setMyHSVTrackbarValue(groupName + "_SIGNAL_RED_TRACKBAR");
                setMyHSVTrackbarValue(groupName + "_SIGNAL_YELLOW_TRACKBAR");
                setMyHSVTrackbarValue(groupName + "_SIGNAL_GREEN_TRACKBAR");
                setMyHSVTrackbarValue(groupName + "_BLUE_AREA_TRACKBAR");;
        }

        ////*Make birdeyeview img*////
        cv::Mat bev;

        bev = src.clone();

        callane.birdEyeView_left(src, parking_white);

        if (groupName == "main")
        { //If you test by video, use one camera (please comment out the other camera)
                if (bird_eye_view)
                {
                        callane.birdEyeView_left(src, bev); //comment out by the other cam
                        if (debug)
                                cv::imshow("bev_le", bev); //comment out by the other cam
                }
                else
                {
                        bev = src.clone();
                }
        }
        else if (groupName == "left")
        {
                callane.birdEyeView_left(src, bev);
                if (debug)
                        cv::imshow("bev_le", bev);
        }
        else if (groupName == "right")
        {

                callane.birdEyeView_right(src, bev);
                if (debug)
                        cv::imshow("bev_ri", bev);
        }

        ////*Detect yellow and white colors and make dst img to binary img by hsv value*////
        if (track_bar)
        { //Use trackbar. Use real-time trackbar's hsv value

                //for detect lane color
                callane.detectYHSVcolor(bev, dst_y, y_hmin, y_hmax, y_smin, y_smax, y_vmin, y_vmax);
                callane.detectWhiteRange(bev, dst_w, w_hmin, w_hmax, w_smin, w_smax, w_vmin, w_vmax, 0, 0);
                callane.detectWhiteRange(parking_white, parking_white, w_hmin, w_hmax, w_smin, w_smax, w_vmin, w_vmax, 0, 0);
                //for detect blocking bar color
                callane.detectWhiteRange(bev, dst_r, r_hmin, r_hmax, r_smin, r_smax, r_vmin, r_vmax, 0, 0);
                //for detect signal lamp
                callane.detectWhiteRange(bev, dst_r2, r2_hmin, r2_hmax, r2_smin, r2_smax, r2_vmin, r2_vmax, 0, 0);
                callane.detectWhiteRange(bev, dst_g, g_hmin, g_hmax, g_smin, g_smax, g_vmin, g_vmax, 0, 0);
                callane.detectWhiteRange(bev, dst_y2, y2_hmin, y2_hmax, y2_smin, y2_smax, y2_vmin, y2_vmax, 0, 0);

                callane.detectWhiteRange(bev, dst_b, b_hmin, b_hmax, b_smin, b_smax, b_vmin, b_vmax, 0, 0);

                cv::imshow(groupName + "_YELLOW_LANE_TRACKBAR", dst_y);
                cv::imshow(groupName + "_WHITE_LANE_TRACKBAR", dst_w);

                cv::imshow(groupName + "_BLOCKING_RED_TRACKBAR", dst_r);

                cv::imshow(groupName + "_SIGNAL_RED_TRACKBAR", dst_r2);
                cv::imshow(groupName + "_SIGNAL_GREEN_TRACKBAR", dst_g);
                cv::imshow(groupName + "_SIGNAL_YELLOW_TRACKBAR", dst_y2);

                cv::imshow(groupName + "_BLUE_AREA_TRACKBAR", dst_b);
        }
        else
        { //Don't use trackbar. Use defalut value.
                callane.detectYHSVcolor(bev, dst_y, 7, 21, 52, 151, 0, 180);
                callane.detectWhiteRange(bev, dst_w, 0, 180, 0, 29, 179, 255, 0, 0);

                callane.detectWhiteRange(bev, dst_r, 160, 179, 0, 29, 179, 255, 0, 0);

                callane.detectWhiteRange(bev, dst_r2, 160, 179, 0, 29, 179, 255, 0, 0);
                callane.detectWhiteRange(bev, dst_g, 38, 75, 0, 29, 179, 255, 0, 0);
                callane.detectWhiteRange(bev, dst_y2, 7, 21, 52, 151, 0, 180, 0, 0);

                callane.detectWhiteRange(bev, dst_b, 53, 114, 152, 255, 64, 255, 0, 0);
        }
}

void InitImgObjectforROS::setProjection(lane_detect_algo::CalLane callane, cv::Mat src, unsigned int *H_aix_Result_color)
{
        ////*Testing histogram*////
        cv::Mat histsrc = src.clone();
        cv::Mat dst = cv::Mat::zeros(histsrc.rows, histsrc.cols, CV_8UC1);
        callane.myProjection(histsrc, dst, H_aix_Result_color);
}

void InitImgObjectforROS::restoreImgWithLangeMerge(lane_detect_algo::CalLane callane, cv::Mat origin_size_img, cv::Mat src_y, cv::Mat src_w, cv::Mat &dst)
{
        ////*Restore birdeyeview img to origin view*////
        cv::Mat laneColor = src_y | src_w;

        dst = origin_size_img.clone();

        if (groupName == "left")
        {
                callane.inverseBirdEyeView_left(laneColor, dst);
        }
        else if (groupName == "right")
        {
                callane.inverseBirdEyeView_right(laneColor, dst);
        }
        else if (groupName == "main")
        { //If you test by video, use one camera (please comment out the other camera)
                if (bird_eye_view)
                {
                        callane.inverseBirdEyeView_left(laneColor, dst); //comment out by the other cam
                        //callane.inverseBirdEyeView_right(laneColor, dst);
                }
                else
                {
                        dst = laneColor.clone();
                }
        }
}

void InitImgObjectforROS::extractLanePoint(cv::Mat origin_src, cv::Mat lane_src)
{
        ////*Make lane infomation msg for translate scan data*////
        cv::Mat output_origin = origin_src.clone();
        cv::Mat pub_img = lane_src.clone();
        int coordi_count = 0;
        coordi_array.data.clear();
        coordi_array.data.push_back(10);
        for (int y = output_origin.rows - 1; y >= 0; y--)
        {
                uchar *origin_data = output_origin.ptr<uchar>(y);
                uchar *pub_img_data;
                if (!gazebo)
                { //for resizing prossesing img to webcam original image(640x320)
                        //pub_img_data = pub_img.ptr<uchar>(y * 0.5); //Restore resize img(0.5 -> 1))
                        pub_img_data = pub_img.ptr<uchar>(y);
                }
                else
                { //use gazebo topic(320x240) - None resize
                        pub_img_data = pub_img.ptr<uchar>(y);
                }
                for (int x = 0; x < output_origin.cols; x++)
                {
                        int temp;
                        if (!gazebo)
                        { //for resizing prossesing img to webcam original image(640x320)
                                //temp = x * 0.5; //Restore resize img(0.5 -> 1)
                                temp = x;
                        }
                        else
                        { //use gazebo topic(320x240) - None resize
                                temp = x;
                        }
                        if (pub_img_data[temp] != (uchar)0)
                        {
                                coordi_count++;
                                coordi_array.data.push_back(x);
                                coordi_array.data.push_back(y);
                                //origin_data[x*output_origin.channels()] = 255;
                                origin_data[x * output_origin.channels() + 1] = 25;
                        }
                }
        }
        coordi_array.data[0] = coordi_count;

        cv::line(output_origin, cv::Point(0, 190 * 2), cv::Point(output_origin.cols - 1, 190 * 2), cv::Scalar(23, 32, 100), 2);
        cv::line(output_origin, cv::Point(0, 200 * 2), cv::Point(output_origin.cols - 1, 200 * 2), cv::Scalar(23, 32, 100), 2);

        cv::imshow(groupName + "_colorfulLane", output_origin);
}

void InitImgObjectforROS::setMySobel(cv::Mat &dst)
{
        if (dst.channels() == 3)
        {
                cv::cvtColor(dst, dst, CV_BGR2GRAY);
        }

        cv::Mat dst_h = dst.clone();
        cv::Mat dst_v = dst.clone();
        cv::Mat element(1, 1, CV_8U, cv::Scalar(1));
        cv::Sobel(dst_h, dst_h, dst_h.depth(), 0, 1); //horizontal
        cv::Sobel(dst_h, dst_v, dst_v.depth(), 1, 0); //vertical
        dst = dst_v;
        cv::dilate(dst, dst, element);
        cv::threshold(dst, dst, 240, 255, cv::THRESH_BINARY);
        cv::medianBlur(dst, dst, 3);
}
void InitImgObjectforROS::setMyLaneBox(cv::Point t_pt, cv::Point b_pt, const string &lane_name, std::vector<cv::Point> &dst)
{
        if (!dst.empty() && lane_name == "left")
        {
                if (dst[1].x > b_pt.x)
                        dst[1].x = b_pt.x;
                if (dst[0].y < t_pt.y)
                        dst[0].y = t_pt.y;
        }
        else if (!dst.empty() && lane_name == "right")
        {
                if (dst[0].x < t_pt.x)
                        dst[0].x = t_pt.x;
                if (dst[0].y < t_pt.y)
                        dst[0].y = t_pt.y;
        }
}
bool InitImgObjectforROS::setMyLaneFitting(cv::Mat &src_img, std::vector<cv::Point> src_pt, const string &lane_name, std::vector<cv::Point> &dst)
{

        bool return_val = true;
        if (lane_name == "left")
        {
                int pre_x_pt = -1, pre_y_pt = -1;
                for (int y = src_pt[1].y; y > src_pt[0].y; y--)
                {
                        uchar *fitting_data = src_img.ptr<uchar>(y);
                        for (int x = src_pt[1].x; x > src_pt[0].x; x--)
                        {
                                if (fitting_data[x] != (uchar)0)
                                {
                                        left_fit_img.ptr<uchar>(y)[x] = (uchar)255;
                                        if (y > 5)
                                        {
                                                int valid_pt = 0;
                                                for (int i = y; i > y - 5; i--)
                                                {
                                                        uchar *pt_test_data = src_img.ptr<uchar>(i);
                                                        if (pt_test_data[x] != (uchar)0)
                                                        {
                                                                valid_pt++;
                                                        }
                                                }
                                                if (valid_pt > 3)
                                                {
                                                        dst.push_back(cv::Point(x, y));
                                                }
                                        }
                                        break;
                                }
                        }
                }
                if (dst.size() <= 10)
                        return_val = false;
        }
        if (lane_name == "right")
        {

                int i = 0;
                int first_fit_y = -1;
                int pre_x_pt = 0, pre_y_pt = 0;
                for (int y = src_pt[1].y; y > src_pt[0].y; y--)
                {
                        uchar *fitting_data = src_img.ptr<uchar>(y);
                        for (int x = src_pt[0].x; x < src_pt[1].x; x++)
                        {
                                if (fitting_data[x] != (uchar)0)
                                {
                                        right_fit_img.ptr<uchar>(y)[x] = (uchar)255;
                                        // if (x == src_pt[0].x)
                                        // {
                                        //         int my_sum = 0;
                                        //         for (int i = x; i < x + 10; i++)
                                        //         {
                                        //                 if (fitting_data[i] != (uchar)0)
                                        //                 {
                                        //                         my_sum++;
                                        //                 }
                                        //         }
                                        //         if (my_sum > 5)
                                        //         {
                                        //                 dst.push_back(cv::Point(x, y));
                                        //                 break;
                                        //                 if (dst.size() > 10)
                                        //                 { ///////////////////////////////////////수정함
                                        //                         return true;
                                        //                 }
                                        //         }
                                        // }
                                        // dst.push_back(cv::Point(x, y));
                                        // break;

                                        if (y > 5)
                                        {
                                                int valid_pt = 0;
                                                for (int i = y; i > y - 5; i--)
                                                {
                                                        uchar *pt_test_data = src_img.ptr<uchar>(i);
                                                        if (pt_test_data[x] != (uchar)0)
                                                        {
                                                                valid_pt++;
                                                        }
                                                }
                                                if (valid_pt > 3)
                                                {
                                                        dst.push_back(cv::Point(x, y));
                                                }
                                        }
                                        break;
                                }
                        }
                }
                if (dst.size() <= 10)
                        return_val = false;
        }
        if (lane_name == "dot_test")
        {
                //int check = 0;
                std::vector<int> lane_width;
                int my_sum = 0, my_avg = 0, my_cnt = 0;
                //saving lane width & inner lane point
                for (int y = src_pt[1].y; y > src_pt[0].y; y--)
                {
                        uchar *fitting_data = src_img.ptr<uchar>(y);
                        for (int x = src_pt[0].x; x < src_pt[1].x; x++)
                        {
                                if (fitting_data[x] != (uchar)0)
                                {
                                        dst.push_back(cv::Point(x, y));
                                        int i = x, width_sum = 0, no_point = 0;
                                        while (no_point < 3)
                                        {
                                                if (i >= src_pt[1].x - 1)
                                                {
                                                        lane_width.push_back(width_sum);
                                                        my_sum += width_sum;
                                                        break;
                                                }
                                                if (fitting_data[i] != (uchar)0)
                                                {
                                                        no_point = 0;
                                                        width_sum++;
                                                        i++;
                                                }
                                                else
                                                {
                                                        no_point++;
                                                        width_sum++;
                                                        i++;
                                                }
                                        }
                                        if (width_sum > 3)
                                        {
                                                lane_width.push_back(width_sum);
                                                my_sum += width_sum;
                                                my_cnt++;
                                        }
                                        break;
                                }
                        }
                }

                if (!lane_width.empty())
                {

                        my_avg = my_sum / lane_width.size();

                        //check lane slope (parking dot lane slope is lete slope)
                        int change_check_y = 0, l_slope = 0, r_slope = 0;
                        for (uint i = 0; i < dst.size(); i++)
                        {
                                //** check dot lane
                                if (dst[i].x >= dst[i + 1].x)
                                {
                                        l_slope++;
                                }
                                else
                                {
                                        r_slope++;
                                }
                        }
                        std::vector<cv::Point> dot_interval;
                        for (uint i = 0; i < dst.size() - 1; i++)
                        {
                                if (abs(dst[i].y - dst[i + 1].y) >= 5 && abs(dst[i].x - dst[i + 1].x) < 17)
                                {

                                        change_check_y++;
                                }
                        }

                        if (change_check_y >= 3)
                        {
                                //checking dot lane width is reliable
                                int reliability = 0;
                                for (int i = 0; i < lane_width.size(); i++)
                                {

                                        if (abs(lane_width[i] - my_avg) < 8)
                                        {
                                                reliability++;
                                        }
                                }
                                if (reliability > lane_width.size() * 0.35)
                                {
                                        return_val = true;
                                }
                                else
                                {
                                        return_val = false;
                                }
                        }
                        else
                        {
                                return_val = false;
                        }
                }
                else
                {
                        return_val = false;
                }
                if (return_val == false && !dst.empty())
                {
                        dst.clear();
                        dst.resize(0);
                }
        }
        return return_val;
}

void InitImgObjectforROS::setMyCanny(const string &lane_name, cv::Mat &dst)
{
        if (dst.channels() == 3)
        {
                cv::cvtColor(dst, dst, CV_BGR2GRAY);
        }
        cv::Canny(dst, dst, (dst.rows + dst.cols) / 4, (dst.rows + dst.cols) / 2);
        cv::medianBlur(dst, dst, 1);
}

void InitImgObjectforROS::setMyMorphology(cv::Mat &dst)
{
        cv::Mat element(3, 3, CV_8U, cv::Scalar(1));
        cv::dilate(dst, dst, element);
        cv::medianBlur(dst, dst, 1);
}

bool InitImgObjectforROS::detectBlockingBar(cv::Mat src)
{
        // std::vector<std::vector<cv::Point>> countours;
        // std::vector<cv::Vec4i> hierachy;
        // cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
        // cv::findContours(src, countours, hierachy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        // cv::Mat draw_lable = cv::Mat::zeros(src.size(), CV_8UC1);
        // for (std::vector<std::vector<cv::Point>>::size_type i = 0; i < countours.size(); ++i)
        // {
        //         cv::drawContours(dst, countours, i, CV_RGB(255, 255, 255), -1, 8, hierachy, 0, cv::Point());
        // }
        // cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY); // Convert the image to Gray
        // cv::threshold(dst, dst, 127, 255, cv::THRESH_BINARY);

        // cv::threshold(dst, draw_lable, 127, 255, cv::THRESH_BINARY_INV);
        // cv::Mat img_labels, stats, centroids;
        // int numOfLables = cv::connectedComponentsWithStats(dst, img_labels, stats, centroids, 8, CV_32S);
        // for (int row = 1; row < numOfLables; row++)
        // {
        //         int *data = stats.ptr<int>(row);
        //         int area = data[cv::CC_STAT_AREA];
        //         int left = data[cv::CC_STAT_LEFT];
        //         int top = data[cv::CC_STAT_TOP];
        //         int width = data[cv::CC_STAT_WIDTH];
        //         int height = data[cv::CC_STAT_HEIGHT];
        //         if (area > 100 && area < 300)
        //         {
        //                 cv::rectangle(draw_lable, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 1);
        //         }
        //         else
        //         {
        //                 cv::rectangle(draw_lable, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 1);
        //                 for (int row = top; row < top + height; row++)
        //                 {
        //                         uchar *data = dst.ptr<uchar>(row);
        //                         for (int col = left; col < left + width; col++)
        //                         {
        //                                 //  data[col] = (uchar)0;
        //                         }
        //                 }
        //         }
        //         cv::putText(draw_lable, std::to_string(area), cv::Point(left + 20, top + 20),
        //                     cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(5, 25, 255), 2);
        //         cv::imshow("red label", draw_lable);
        //         cv::imshow("red data", dst);
        // }
        // int first_red = -1, last_red = -1, red_sum = 0, no_pt = 0, red_cnt = 0, stop = 0;

        cv::Mat blocking_bar_mask = cv::Mat::zeros(src.size(), CV_8UC1);
        for (int y = 30; y < 80; y++)
        {
                uchar *red_mask_data = blocking_bar_mask.ptr<uchar>(y);
                for (int x = 0; x < blocking_bar_mask.cols; x++)
                {
                        red_mask_data[x] = (uchar)255;
                }
        }
        //cv::imshow("blocking_bar_mask",blocking_bar_mask);
        cv::Mat blocking_bar_roi = blocking_bar_mask & src;
        int red_cnt = 0;
        for (int y = 30; y < 80; y++)
        {
                uchar *red_cnt_data = blocking_bar_roi.ptr<uchar>(y);
                for (int x = 0; x < blocking_bar_roi.cols; x++)
                {
                        if (red_cnt_data[x] != (uchar)0)
                        {
                                red_cnt++;
                        }
                }
        }
        //std::cout<<"red cnt : "<<red_cnt<<std::endl;
        //cv::imshow("blocking_bar_roi",blocking_bar_roi);

        if (red_cnt > blocking_bar_roi.cols * (80 - 60) * 0.1)
        {
                return true;
        }
        else
        {
                return false;
        }
        // for (int y = 0; y < 80; y++)
        // { //blockbar roi
        //         uchar *red_data = src.ptr<uchar>(y);
        //         no_pt = 0;
        //         for (int x = 0; x < src.cols; x++)
        //         {
        //                 if (red_data[x] != 0)
        //                 {
        //                         if (no_pt <= 5)
        //                                 no_pt = 0;
        //                         red_sum++;
        //                         if (red_sum > 20)
        //                         {
        //                                 red_sum = 0;
        //                                 no_pt = 0;
        //                                 for (int i = x + 1; i < src.cols; i++)
        //                                 {
        //                                         if (red_data[i] != 0)
        //                                         {
        //                                                 red_sum++;
        //                                                 if (red_sum > 20 && no_pt > 20)
        //                                                 {
        //                                                         red_cnt++;
        //                                                         red_sum = 0;
        //                                                         no_pt = 0;
        //                                                         if (red_cnt > 2)
        //                                                         {
        //                                                                 stop = 1; //detect blocking bar
        //                                                                 break;
        //                                                         }
        //                                                 }
        //                                         }
        //                                         else
        //                                         {
        //                                                 no_pt++;
        //                                         }
        //                                 }
        //                                 break;
        //                         }
        //                 }
        //                 else
        //                 {
        //                         no_pt++;
        //                         if (no_pt > 5)
        //                                 break;
        //                 }
        //         }
        //         if (stop == 1)
        //                 break;
        // }
        // if (stop == 1)
        //         return true;
        // else
        //      return false;
}

///***two arg***///
void InitImgObjectforROS::setRoi(const string &lane_name, cv::Mat &dst)
{
        if (dst.channels() == 3)
        {
                cv::cvtColor(dst, dst, CV_BGR2GRAY);
        }

        if (lane_name == "left")
        {

                ////*for delete bias that calibrated img(because this bias has none data)*////
                for (int y = 0; y < dst.rows / 2; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = 0; x < dst.cols; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }

                for (int y = 0; y < dst.rows; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = 0; x < 11; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
        }
        else if (lane_name == "right")
        {

                for (int y = 0; y < dst.rows / 2; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = 0; x < dst.cols; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                // for (int y = dst.rows - 3; y < dst.rows; y++)

                for (int y = 0; y < dst.rows; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = dst.cols - 10; x < dst.cols - 1; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
        }
}

///***three arg***///
void InitImgObjectforROS::setRoi(const string &lane_name, cv::Mat &dst, std::vector<cv::Point> &pt_dst)
{
        if (dst.channels() == 3)
        {
                cv::cvtColor(dst, dst, CV_BGR2GRAY);
        }

        if (lane_name == "left")
        {

                cv::Point left_roi_t(0, dst.rows / 2);
                // cv::Point left_roi_b(dst.cols / 2 -1, dst.rows - 3);
                cv::Point left_roi_b((dst.cols / 2) + 50, dst.rows - 1);
                pt_dst.push_back(left_roi_t);
                pt_dst.push_back(left_roi_b);

                ////*for delete bias that calibrated img(because this bias has none data)*////
                for (int y = 0; y < dst.rows / 2; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = 0; x < dst.cols; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for (int y = dst.rows / 2 - 1; y < dst.rows; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = left_roi_b.x; x < dst.cols; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                // for (int y = 0; y < dst.rows; y++)
                // {
                //         uchar *none_roi_data = dst.ptr<uchar>(y);
                //         for (int x = dst.cols / 2; x < dst.cols; x++)
                //         {
                //                 if (none_roi_data[x] != (uchar)0)
                //                 {
                //                         none_roi_data[x] = (uchar)0;
                //                 }
                //         }
                // }
                // for (int y = 0; y < dst.rows; y++)
                // {
                //         uchar *none_roi_data = dst.ptr<uchar>(y);
                //         for (int x = 0; x < 11; x++)
                //         {
                //                 if (none_roi_data[x] != (uchar)0)
                //                 {
                //                         //none_roi_data[x] = (uchar)0;
                //                 }
                //         }
                // }
        }
        else if (lane_name == "right")
        {
                // cv::Point right_roi_t(dst.cols / 2 + 1, dst.rows / 2);
                cv::Point right_roi_t((dst.cols / 2) - 50, dst.rows / 2);
                cv::Point right_roi_b(dst.cols - 1, dst.rows - 1);
                pt_dst.push_back(right_roi_t);
                pt_dst.push_back(right_roi_b);

                for (int y = 0; y < dst.rows / 2; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = 0; x < dst.cols; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                for (int y = dst.rows / 2 - 1; y < dst.rows; y++)
                {
                        uchar *none_roi_data = dst.ptr<uchar>(y);
                        for (int x = 0; x < right_roi_t.x; x++)
                        {
                                if (none_roi_data[x] != (uchar)0)
                                {
                                        none_roi_data[x] = (uchar)0;
                                }
                        }
                }
                // for (int y = 0; y < dst.rows; y++)
                // {
                //         uchar *none_roi_data = dst.ptr<uchar>(y);
                //         for (int x = 0; x <  dst.cols / 2; x++)
                //         {
                //                 if (none_roi_data[x] != (uchar)0)
                //                 {
                //                         none_roi_data[x] = (uchar)0;
                //                 }
                //         }
                // }
                // for (int y = 0; y < dst.rows; y++)
                // {
                //         uchar *none_roi_data = dst.ptr<uchar>(y);
                //         for (int x = dst.cols - 10; x < dst.cols - 1; x++)
                //         {
                //                 if (none_roi_data[x] != (uchar)0)
                //                 {
                //                         //none_roi_data[x] = (uchar)0;
                //                 }
                //         }
                // }
        }
}

bool InitImgObjectforROS::signalRedDetection(cv::Mat src_red)
{
        cv::Mat red_sign_mask = cv::Mat::zeros(src_red.size(), CV_8UC1);
        for (int y = 40; y < 120; y++)
        {
                uchar *red_sign_mask_data = red_sign_mask.ptr<uchar>(y);
                for (int x = 200; x < 250; x++)
                {
                        red_sign_mask_data[x] = (uchar)255;
                }
        }

        cv::Mat red_sign_roi = red_sign_mask & src_red;
        int red_cnt = 0;
        for (int y = 40; y < 120; y++)
        {
                uchar *red_cnt_data = red_sign_roi.ptr<uchar>(y);
                for (int x = 200; x < 250; x++)
                {
                        if (red_cnt_data[x] != (uchar)0)
                        {
                                red_cnt++;
                        }
                }
        }
        std::cout << "***** Red signal detection *****" << std::endl;
        std::cout << "red signal cnt : " << red_cnt << std::endl;

        if (red_cnt > 20 && red_cnt < 60)
        {
                return true;
        }
        else
        {
                return false;
        }
}
bool InitImgObjectforROS::signalGreenDetection(cv::Mat src_green)
{

        cv::Mat green_sign_mask = cv::Mat::zeros(src_green.size(), CV_8UC1);
        for (int y = 90; y < 170; y++)
        {
                uchar *green_sign_mask_data = green_sign_mask.ptr<uchar>(y);
                for (int x = 200; x < 250; x++)
                {
                        green_sign_mask_data[x] = (uchar)255;
                }
        }

        cv::Mat green_sign_roi = green_sign_mask & src_green;
        int green_cnt = 0;
        for (int y = 90; y < 170; y++)
        {
                uchar *green_cnt_data = green_sign_roi.ptr<uchar>(y);
                for (int x = 200; x < 250; x++)
                {
                        if (green_cnt_data[x] != (uchar)0)
                        {
                                green_cnt++;
                        }
                }
        }
        std::cout << "***** Green signal detection *****" << std::endl;
        std::cout << "green signal cnt : " << green_cnt << std::endl;
        if (green_cnt > 25)
        {
                return true;
        }
        else
        {
                return false;
        }
}
bool InitImgObjectforROS::signalYellowDetection(cv::Mat src_yellow)
{

        cv::Mat yellow_sign_mask = cv::Mat::zeros(src_yellow.size(), CV_8UC1);
        for (int y = 50; y < 120; y++)
        {
                uchar *yellow_sign_mask_data = yellow_sign_mask.ptr<uchar>(y);
                for (int x = 200; x < 250; x++)
                {
                        yellow_sign_mask_data[x] = (uchar)255;
                }
        }

        cv::Mat yellow_sign_roi = yellow_sign_mask & src_yellow;
        int yellow_cnt = 0;
        for (int y = 50; y < 120; y++)
        {
                uchar *yellow_cnt_data = yellow_sign_roi.ptr<uchar>(y);
                for (int x = 200; x < 250; x++)
                {
                        if (yellow_cnt_data[x] != (uchar)0)
                        {
                                yellow_cnt++;
                        }
                }
        }
        cv::imshow("yellow roi", yellow_sign_roi);
        std::cout << "***** Yellow signal detection *****" << std::endl;
        std::cout << "yellow signal cnt : " << yellow_cnt << std::endl;
        if (yellow_cnt > 30 && yellow_cnt < 70)
        {
                return true;
        }
        else
        {
                return false;
        }
}

int InitImgObjectforROS::checkDirectionWithFlann(cv::Mat &src, int min_hessian)
{
        int return_val = 0;

        ///**https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html**//
        cv::Mat left_turn_src = cv::imread("/home/seuleee/Pictures/left_turn_t1.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat right_turn_src = cv::imread("/home/seuleee/Pictures/right_turn_t2.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat input_src = src.clone();
        cv::cvtColor(input_src, input_src, CV_BGR2GRAY);
        cv::Mat descriptors_input, descriptors_left_turn, descriptors_right_turn;

        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(min_hessian);
        std::vector<cv::KeyPoint> keypoints_input, keypoints_left_turn, keypoints_right_turn;
        
        detector->detectAndCompute(input_src, cv::noArray(), keypoints_input, descriptors_input);               
        detector->detectAndCompute(left_turn_src, cv::noArray(), keypoints_left_turn, descriptors_left_turn);
        detector->detectAndCompute(right_turn_src, cv::noArray(), keypoints_right_turn, descriptors_right_turn);

        

        cv::Ptr<cv::DescriptorMatcher> matcher1 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cv::Ptr<cv::DescriptorMatcher> matcher2 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches_left, knn_matches_right;
        if(descriptors_input.size >= descriptors_left_turn.size){
                matcher1->knnMatch(descriptors_left_turn, descriptors_input, knn_matches_left, 2);
        }
        else{
                matcher1->knnMatch(descriptors_input,descriptors_left_turn,  knn_matches_left, 2);
        }

        if(descriptors_input.size >= descriptors_right_turn.size){
                matcher2->knnMatch(descriptors_right_turn, descriptors_input, knn_matches_right, 2);
        }
        else{
                matcher2->knnMatch(descriptors_input,descriptors_right_turn,  knn_matches_right, 2);
        }


        const float ratio_thresh = 0.7f;
        if(for_gui){
                std::vector<cv::DMatch> good_matches_left, good_matches_right;
                for (size_t i = 0; i < knn_matches_left.size(); i++)
                {
                        if (knn_matches_left[i][0].distance < ratio_thresh * knn_matches_left[i][1].distance)
                        {
                                good_matches_left.push_back(knn_matches_left[i][0]);
                        }
                }
                for (size_t i = 0; i < knn_matches_right.size(); i++)
                {
                        if (knn_matches_right[i][0].distance < ratio_thresh * knn_matches_right[i][1].distance)
                        {
                                good_matches_right.push_back(knn_matches_right[i][0]);
                        }
                }
                cv::Mat img_matches;
                if(good_matches_left.size() > good_matches_right.size()){

                        if(descriptors_input.size >= descriptors_left_turn.size){
                               cv::drawMatches(left_turn_src, keypoints_left_turn, input_src, keypoints_input, good_matches_left, img_matches,
                                               cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        }
                        else{
                                cv::drawMatches(input_src, keypoints_input, left_turn_src, keypoints_left_turn, good_matches_left, img_matches,
                                               cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        }
                        // cv::drawMatches(left_turn_src, keypoints_left_turn, input_src, keypoints_input, good_matches_left, img_matches,
                        // cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        if(good_matches_left.size() > 3){
                                return_val = -1;
                                printf("*******************************************************left turn !!!!\n");   
                        }
                        else{
                                return_val = 0;
                        }
                        printf("good match left size : %d\n",good_matches_left.size());
                        cv::imshow("surf keypoints", img_matches);
                }
                else if(good_matches_left.size() < good_matches_right.size()){
                        if(descriptors_input.size >= descriptors_right_turn.size){
                               cv::drawMatches(right_turn_src, keypoints_right_turn, input_src, keypoints_input, good_matches_right, img_matches,
                                               cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        }
                        else{
                                cv::drawMatches(input_src, keypoints_input, right_turn_src, keypoints_right_turn, good_matches_right, img_matches,
                                               cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        }
                        
                        // cv::drawMatches(right_turn_src, keypoints_right_turn, input_src, keypoints_input,  good_matches_right, img_matches,
                        // cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        if(good_matches_right.size()> 3){
                                return_val = 1;
                                printf("*******************************************************right turn !!!!\n");
                        }
                        else{
                                return_val = 0;
                        }
                        printf("good match right size : %d\n",good_matches_right.size());
                        cv::imshow("surf keypoints", img_matches);
                }
                else{
                        printf("Both good matches size that left turn and right turn is same...\n");
                }
                
                
        }
        else{
                int left_turn_score = 0, right_turn_score = 0;
                for (size_t i = 0; i < knn_matches_left.size(); i++)
                {
                        if (knn_matches_left[i][0].distance < ratio_thresh * knn_matches_left[i][1].distance)
                        {
                                left_turn_score++;       
                        }
                }
                for (size_t i = 0; i < knn_matches_right.size(); i++)
                {
                        if (knn_matches_right[i][0].distance < ratio_thresh * knn_matches_right[i][1].distance)
                        {
                                right_turn_score++;
                        }
                }
                if(left_turn_score > right_turn_score){
                        if(left_turn_score > 4){
                                return_val = -1;
                        }
                        else{
                                return_val = 0;
                        }
                }
                else if(left_turn_score < right_turn_score){
                        if(right_turn_score>4){
                                return_val = 1;
                        }
                        else{
                                return_val = 0;
                        }
                }
                else{
                        printf("Both good matches size that left turn and right turn is same...\n");
                }
        }
        
        return return_val;

        // ///**https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html**//
        // cv::Mat left_turn_src = cv::imread("/home/seuleee/Pictures/left_turn_t1.jpg",cv::IMREAD_GRAYSCALE);
        // cv::Mat right_turn_src = cv::imread("/home/seuleee/Pictures/right_turn_t2.jpg",cv::IMREAD_GRAYSCALE);
        // int min_hessian = 400;

        // cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(min_hessian);
        // std::vector<cv::KeyPoint> keypoints1, keypoints2;
        // cv::Mat descriptors1, descriptors2;
        // cv::Mat left_turn_origin = frame.clone();
        // cv::cvtColor(left_turn_origin,left_turn_origin,CV_BGR2GRAY);
        // detector->detectAndCompute(left_turn_src, cv::noArray(),keypoints1,descriptors1);
        // detector->detectAndCompute(left_turn_origin, cv::noArray(),keypoints2,descriptors2);

        // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // std::vector<std::vector<cv::DMatch>> knn_matches;
        //if(descriptors2.size > descriptors1.size)
        // matcher->knnMatch(descriptors1,descriptors2,knn_matches,2);
        //else
        // matcher->knnMatch(descriptors2,descriptors1,knn_matches,2);
        // const float ratio_thresh = 0.7f;
        // std::vector<cv::DMatch> good_matches;
        // for(size_t i = 0; i< knn_matches.size(); i++){
        //         if(knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
        //                 good_matches.push_back(knn_matches[i][0]);
        //         }
        // }
        // cv::Mat img_matches;
        //if(left_turn_src.size > left_turn_origin)
        // cv::drawMatches(left_turn_origin,keypoints2,left_turn_src,keypoints1,good_matches,img_matches,
        //                 cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //else
        // cv::drawMatches(left_turn_src,keypoints1,left_turn_origin,keypoints2,good_matches,img_matches,
        //                 cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // std::cout<<"good matches size : "<<good_matches.size()<<std::endl;
        // cv::imshow("surf keypoints", img_matches);
        // ///**https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html**//
}
bool InitImgObjectforROS::checkParkingWithFlann(cv::Mat &src, int min_hessian){
        bool return_val = false;

        ///**https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html**//
        cv::Mat parking_src = cv::imread("/home/seuleee/Pictures/parking.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat input_src = src.clone();
        cv::cvtColor(input_src, input_src, CV_BGR2GRAY);
        cv::Mat descriptors_input, descriptors_parking;

        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(min_hessian);
        std::vector<cv::KeyPoint> keypoints_input, keypoints_parking;
        
        detector->detectAndCompute(input_src, cv::noArray(), keypoints_input, descriptors_input);               
        detector->detectAndCompute(parking_src, cv::noArray(), keypoints_parking, descriptors_parking);
       

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        if(descriptors_input.size > descriptors_parking.size){
                matcher->knnMatch(descriptors_parking, descriptors_input, knn_matches, 2);
        }
        else{
                matcher->knnMatch(descriptors_input, descriptors_parking, knn_matches, 2);
        }

        const float ratio_thresh = 0.7f;
        if(for_gui){
                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < knn_matches.size(); i++)
                {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                        {
                                good_matches.push_back(knn_matches[i][0]);
                        }
                }
                
                cv::Mat img_matches;
                if(good_matches.size() > 4){
                        if(input_src.size > parking_src.size){
                                cv::drawMatches(parking_src, keypoints_parking, input_src, keypoints_input, good_matches, img_matches,
                                                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        }
                        else{
                                cv::drawMatches(input_src, keypoints_input, parking_src, keypoints_parking,  good_matches, img_matches,
                                                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        }
                        return_val = true;
                        printf("*******************************************************parking area !!!!\n");   
                        printf("good match parking size : %d\n",good_matches.size());
                        cv::imshow("surf keypoints", img_matches);
                }
                else{
                        return_val = false;
                        
                }       
        }
        else{
                int parking_score = 0;
                for (size_t i = 0; i < knn_matches.size(); i++)
                {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                        {
                                parking_score++;       
                        }
                }
           
                if(parking_score > 4){
                        return_val = true;
                }
                else {
                        return_val = false;
                }
                
        }
        
        return return_val;
}

bool InitImgObjectforROS::checkTunnelWithFlann(cv::Mat &src, int min_hessian){
        bool return_val = false;

        ///**https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html**//
        cv::Mat parking_src = cv::imread("/home/seuleee/Pictures/tunnel.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat input_src = src.clone();
        cv::cvtColor(input_src, input_src, CV_BGR2GRAY);
        cv::Mat descriptors_input, descriptors_tunnel;

        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(min_hessian);
        std::vector<cv::KeyPoint> keypoints_input, keypoints_tunnel;
        
        detector->detectAndCompute(input_src, cv::noArray(), keypoints_input, descriptors_input);               
        detector->detectAndCompute(parking_src, cv::noArray(), keypoints_tunnel, descriptors_tunnel);
       

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors_tunnel, descriptors_input, knn_matches, 2);

        const float ratio_thresh = 0.7f;
        if(for_gui){
                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < knn_matches.size(); i++)
                {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                        {
                                good_matches.push_back(knn_matches[i][0]);
                        }
                }
                
                cv::Mat img_matches;
                if(good_matches.size() > 4){
                        cv::drawMatches(parking_src, keypoints_tunnel, input_src, keypoints_input, good_matches, img_matches,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                        return_val = true;
                        printf("*******************************************************tunnel enter !!!!\n");   
                        printf("good match tunnel size : %d\n",good_matches.size());
                        cv::imshow("surf keypoints", img_matches);
                }
                else{
                        return_val = false;
                        
                }       
        }
        else{
                int parking_score = 0;
                for (size_t i = 0; i < knn_matches.size(); i++)
                {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                        {
                                parking_score++;       
                        }
                }
           
                if(parking_score > 4){
                        return_val = true;
                }
                else {
                        return_val = false;
                }
                
        }
        
        return return_val;
}
bool InitImgObjectforROS::checkBlueArea(cv::Mat &src, cv::Point &left_top_blue, cv::Point &right_bottom_blue){
        std::vector<std::vector<cv::Point>> countours;
        std::vector<cv::Vec4i> hierachy;

        bool return_val = false;
        
        cv::findContours(src, countours, hierachy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

        for (std::vector<std::vector<cv::Point>>::size_type i = 0; i < countours.size(); ++i)
        {
                cv::drawContours(dst, countours, i, CV_RGB(255, 255, 255), -1, 8, hierachy, 0, cv::Point());
        }
        cv::threshold(dst, dst, 127, 255, cv::THRESH_BINARY);
        
        cv::Mat draw_lable;
        cv::threshold(dst, draw_lable, 127, 255, cv::THRESH_BINARY_INV);

        cv::Mat img_labels, stats, centroids;
        int numOfLables = cv::connectedComponentsWithStats(dst, img_labels, stats, centroids, 8, CV_32S);

        
        for (int row = 1; row < numOfLables; row++)
        {

                int *data = stats.ptr<int>(row);
                int area = data[cv::CC_STAT_AREA];
                int left = data[cv::CC_STAT_LEFT];
                int top = data[cv::CC_STAT_TOP];
                int width = data[cv::CC_STAT_WIDTH];
                int height = data[cv::CC_STAT_HEIGHT];
                // cv::rectangle(draw_lable, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 1);
                // cv::putText(draw_lable, std::to_string(area), cv::Point(left + 20, top + 20),
                //             FONT_HERSHEY_SIMPLEX, 0.3, Scalar(5, 25, 255), 2);
                // cv::imshow("w_lable", draw_lable);
                
                if (width >= 20 && height >= 20 && abs(height-width) <10)
                {
                        
                        left_top_blue.x = left;
                        left_top_blue.y = top;
                        right_bottom_blue.x = left+width;
                        right_bottom_blue.y = top+height;
                        return_val = true;
                        return return_val;
                }
                
                if(for_gui){
                        cv::rectangle(draw_lable, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 1);
                        cv::putText(draw_lable, std::to_string(width), cv::Point(left + 20, top - 20),
                                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(5, 25, 255), 1);
                        cv::putText(draw_lable, std::to_string(height), cv::Point(left, top - 20),
                                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(5, 25, 255), 1);
                        cv::imshow("blue_lable", draw_lable);
                }
                
               
        }
        return return_val;
         
}
int main(int argc, char **argv)
{
        ros::init(argc, argv, "lane_detection");
        if (!gazebo)
        { //if you subscribe topic that published camera_image pkg
                groupName = argv[1];
        }
        else
        {                           //if you use another img topic
                groupName = "main"; //for test pointxyzrgb to mat (/camera/depth_registerd/points) please set "light" or "right"
        }

        ROS_INFO("strat lane detection");
        InitImgObjectforROS img_obj;
        ros::Rate loop_rate(45);
        //record flag 만들기, group node launch파일에 복구
        if (auto_record)
        {
                if (groupName == "left")
                        video_left.open("/home/seuleee/Desktop/autorace_video_src/0717/left_record.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(640 / 2, 480 / 2), isColor);
                else if (groupName == "right")
                        video_right.open("/home/seuleee/Desktop/autorace_video_src/0717/right_record.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(640 / 2, 480 / 2), isColor);
                else if (groupName == "main")
                        video_main.open("/home/seuleee/Desktop/autorace_video_src/1110/tunnel3.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(640 / 2, 480 / 2), isColor);
        }

        while (img_obj.nh.ok())
        {
                img_obj.pub.publish(img_obj.coordi_array);
                img_obj.ang_vel_pub.publish(img_obj.goal_array);
                img_obj.goal_pub.publish(img_obj.goal_array);

                img_obj.reset_msg_pub.publish(img_obj.reset_val);
                ros::spinOnce();
                loop_rate.sleep();
        }
        ROS_INFO("program killed!\n");
        return 0;
}
