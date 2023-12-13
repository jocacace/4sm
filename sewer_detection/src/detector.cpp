#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "boost/thread.hpp"
#include "sensor_msgs/CameraInfo.h"
#include <ros/package.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <mutex>
#include <opencv2/features2d.hpp>
#include "Eigen/Dense"
#include "std_srvs/Empty.h"
#include "sewer_detection/detections.h"
#include "tf/transform_broadcaster.h"
#include "tf/transform_listener.h"
#include "geometry_msgs/Pose2D.h"
#include "std_msgs/Int32.h"
#include "geometry_msgs/Point.h"

using namespace std;
using namespace cv; 

#define TEST_GOTO (0)

std::mutex img_mutex;

typedef struct ellipses {
    Eigen::Vector2i center;
    Eigen::Vector3d real_pos;
    float circularity;
    int itr = 0;
    int id;
    RotatedRect shape;
    bool visible;
}ellipses;


class SewerDetector {
    public:
        SewerDetector();
        void run();
        void img_cb( const sensor_msgs::Image & img );
        void elaborate();
        void get_ellipse( const Mat & img, vector< Point> & ellips, Point & center );
        bool start_stop_detector(std_srvs::Empty::Request  &req, std_srvs::Empty::Response &res);
        bool shutdown_detector(std_srvs::Empty::Request  &req, std_srvs::Empty::Response &res);
        void test_goto();

        void get_destination_point_cb(const geometry_msgs::Pose2D & d);
        void get_sewer_destination_cb(const std_msgs::Int32 & d);
        Eigen::Vector3d calc_destination_point(const int px, const int py);



    private:
        ros::NodeHandle _nh;
        ros::Subscriber _img_sub;
        ros::ServiceServer _start_stop_detector_srv;
        ros::ServiceServer _shutdown_detector_srv;
        ros::Subscriber _sewer_destination_sub;
        ros::Subscriber _point_destination_sub;
        ros::Publisher _detections_pub;
        ros::Publisher _motion_control_destination_pub;
        Mat _src;
        bool _img_ready;
        vector< ellipses > _ellipses;
        int _last_id;
        int _ellipses_distance;
        int _min_itr;
        int _min_px_area;
        int _resize_scale_factor;
        bool _to_detect;
        bool _to_exit;
        tf::TransformListener _listener;
        tf::StampedTransform  _st;

        double _K_00;
        double _K_02;
        double _K_12;
        double _K_11;

        bool _new_destination_point;
        bool _new_sewer_destination;
        
        geometry_msgs::Pose2D _destination_point;

};


inline Eigen::Matrix3d QuatToMat(Eigen::Vector4d Quat){
    Eigen::Matrix3d Rot;
    float s = Quat[0];
    float x = Quat[1];
    float y = Quat[2];
    float z = Quat[3];
    Rot << 1-2*(y*y+z*z),2*(x*y-s*z),2*(x*z+s*y),
    2*(x*y+s*z),1-2*(x*x+z*z),2*(y*z-s*x),
    2*(x*z-s*y),2*(y*z+s*x),1-2*(x*x+y*y);
    return Rot;
}

SewerDetector::SewerDetector() {

    string image_topic;
    if( !_nh.getParam( "img_tpoic", image_topic)) {
        image_topic = "tracking";
    }
    if( !_nh.getParam( "ellipses_distance", _ellipses_distance)) {
        _ellipses_distance = 50;
    }
    if( !_nh.getParam( "min_itr", _min_itr)) {
        _min_itr = 50;
    }
    if( !_nh.getParam( "min_px_area", _min_px_area)) {
        _min_px_area = 1500;
    }
    if( !_nh.getParam( "resize_scale_factor", _resize_scale_factor)) {
        _resize_scale_factor = 1;
    }
    string cam_info_topic;
    if( !_nh.getParam("cam_info_topic", cam_info_topic)) {
        cam_info_topic = "/iris/tracking_camera_45/camera_info";
    }

    if( !_nh.getParam("K_00", _K_00)) {
        _K_00 = 139.1395614431696 ;
    }
    if( !_nh.getParam("K_02", _K_02)) {
        _K_02 = 320.5;
    }
    if( !_nh.getParam("K_12", _K_12)) {
        _K_12 = 240.5;
    }
    if( !_nh.getParam("K_11", _K_11)) {
        _K_11 = 139.1395614431696;
    }

    



    _sewer_destination_sub = _nh.subscribe("/detector_client/selected_alcantarilla", 1, &SewerDetector::get_sewer_destination_cb, this);
    _point_destination_sub = _nh.subscribe("/detector_client/selected_point", 1, &SewerDetector::get_destination_point_cb, this);

    _motion_control_destination_pub = _nh.advertise<geometry_msgs::Point>("/motion_client/target_point", 1);
    _detections_pub = _nh.advertise<sewer_detection::detections>("/sewer_detection/detections", 1);
    _start_stop_detector_srv = _nh.advertiseService("start_stop_sewer_detector", &SewerDetector::start_stop_detector, this);
    _shutdown_detector_srv = _nh.advertiseService("shutdown_sewer_detector", &SewerDetector::shutdown_detector, this);
    _img_sub = _nh.subscribe( image_topic, 1, &SewerDetector::img_cb, this);
    _img_ready = false;
    _last_id = 0;
    _to_detect = true;
    _to_exit = false;
    _new_destination_point = false;
    _new_sewer_destination = false;

}


void SewerDetector::get_destination_point_cb(const geometry_msgs::Pose2D & d) {
    _destination_point = d;
    _new_destination_point = true;
}
void SewerDetector::get_sewer_destination_cb(const std_msgs::Int32 & d) {
    
    _new_sewer_destination = true;
}


int find_closest_ellipse( const Eigen::Vector2i & c, const vector<ellipses> & ell, const int & min_dist_param) {

    bool found = false;
    int index = 0;
    int e = -1;
    while( !found && index < ell.size() ) {
        //cout << "Distance: " << index << ": " << (ell[index].center - c).norm() << endl; 
        if(  (ell[index].center - c).norm() < min_dist_param ) {
            found = true;
            e = index;
        }
        else 
            index++;
    }
    return e;
}


bool SewerDetector::shutdown_detector(std_srvs::Empty::Request  &req, std_srvs::Empty::Response &res) {
  _to_exit = true;
  return true;
}

bool SewerDetector::start_stop_detector(std_srvs::Empty::Request  &req, std_srvs::Empty::Response &res) {
  _to_detect = !_to_detect;
  return true;
}

void SewerDetector::img_cb( const sensor_msgs::Image & img ) {
	cv_bridge::CvImagePtr cv_ptr;
	try {
		cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        
        img_mutex.lock();
        _src = cv_ptr->image;
        img_mutex.unlock();

        _img_ready = true;
	}
	catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
}


void SewerDetector::get_ellipse( const Mat & img_in, vector< Point> & ellips, Point & center ) {    
    
    int ellipse_min_c[2];
    ellipse_min_c[0] = ellipse_min_c[1] = 1000;
    RotatedRect minEllipse;
    
    Mat img;
    img_in.copyTo( img );

    Mat out_img;
    Mat canny;
    Mat gray;    
    vector<vector<Point> > contours;

    int down_height = img.rows / _resize_scale_factor;
    int down_width = img.cols  / _resize_scale_factor;
    resize(img, img, Size(down_width, down_height), INTER_LINEAR);
    //undistort(img, img, _cam_cameraMatrix, _cam_distCo);
    
    GaussianBlur( img, img, Size(3,3), 1, 1 );
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 29, 10);
    morphologyEx(gray, gray, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, cv::Size(5, 5)));
    dilate(gray, gray, Mat(), Point(-1, -1), 3, 1, 1);
    cv::Canny(gray,gray,300, 200);
    
    findContours(gray, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(img.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ ) {
        Scalar color = Scalar( 0, 0, 255 );
        drawContours( drawing, contours, i, color, 2, 8 );
    }
    //imshow( "contours", drawing );
    drawing.release();
    //waitKey(1);

    vector<Moments> mu(contours.size() );
    img.copyTo( out_img );

    for(int i=0; i<_ellipses.size(); i++) {
        //delete only after serveral times unseen
        _ellipses[i].visible = false;
    }


    for(int e=0; e<contours.size(); e++) {
        if( contourArea(contours[e]) > _min_px_area ) {

            minEllipse = fitEllipse( Mat(contours[e]) );
            center = minEllipse.center;

            Eigen::Vector2i c;
            c << center.x, center.y;
            int e_index = find_closest_ellipse ( c, _ellipses, _ellipses_distance );
            if( e_index > -1 ) {
                _ellipses[e_index].itr++;
                _ellipses[e_index].center = c;
                _ellipses[e_index].shape = minEllipse;
                _ellipses[e_index].visible = true;
            }
            else {
                ellipses e;
                e.center = c;
                e.circularity = 0; //Todo calculate
                e.itr = 0;
                e.id = _last_id;
                _last_id++;
                e.shape = minEllipse;
                _ellipses.push_back( e );
            }

            //cout << "center: " << center.x << " " << center.y << endl;
            //ellips = contours[e];
            //Scalar color = Scalar( 0, 0, 255 );
            //ellipse( out_img, minEllipse, color, 2, 8 );
            //circle(  out_img, minEllipse.center, 2, color, 2, 8 );
        }
    }

    cout << "---------------------Ellipses list--------------------" << endl;
    cout << "Size: " << _ellipses.size() << endl;
    for(int i=0; i<_ellipses.size(); i++) {
        cout << "Ellipse: [" << i << "]: id - " << _ellipses[i].id << endl;
        cout << "Ellipse: [" << i << "]: Center - " << _ellipses[i].center.transpose() << endl;
        cout << "Ellipse: [" << i << "]: Circularity - " << _ellipses[i].circularity << endl;        
        cout << "Ellipse: [" << i << "]: Itr - " << _ellipses[i].itr << endl;        
    }
    cout << "------------------------------------------------------" << endl;


    sewer_detection::detections dets;
    sewer_detection::detection det;
    for(int i=0; i<_ellipses.size(); i++) {

        if( _ellipses[i].itr > _min_itr ) {
            
            
            Scalar color = Scalar( 0, 0, 255 );
            ellipse( out_img, _ellipses[i].shape, color, 2, 8 );
            //circle(  out_img, minEllipse.center, 2, color, 2, 8 );
            cv::putText(out_img, //target image
                    std::to_string( _ellipses[i].id ), //text
                    cv::Point(     _ellipses[i].center[0], _ellipses[i].center[1]      ), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(118, 185, 0), //font color
                    2);
            
            
            det.id = _ellipses[i].id;
            det.point.x = _ellipses[i].center[0];
            det.point.y = _ellipses[i].center[1];
            dets.sewers.push_back( det );
        }
    }

    _detections_pub.publish( dets );

    for(int i=0; i<_ellipses.size(); i++) {
        if( _ellipses[i].visible == false ) {
            _ellipses.erase( _ellipses.begin() + i);
        }
    } //Remove unseen ellipses

    //imshow( "gray", gray );
    //imshow("output", out_img);
    //waitKey(1);
    contours.clear();

}


Eigen::Vector3d SewerDetector::calc_destination_point(const int px, const int py) {
    
    static tf::TransformBroadcaster br;  
    tf::Transform transform;      
    tf::Quaternion q0;  
    q0.setRPY(0, 0, 0.0);  


    _listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(0.1));
    try {     
        _listener.lookupTransform("map", "base_link", ros::Time(0), _st);
    }    
    catch (tf::TransformException ex) {     
        ROS_ERROR("%s",ex.what());      
    }
    
    Eigen::Vector3d base_link_pos;
    base_link_pos << _st.getOrigin().x(), _st.getOrigin().y(), _st.getOrigin().z();
    _listener.waitForTransform("map", "tracking_camera_45_link_optical", ros::Time(0), ros::Duration(0.1));
    try {     
        _listener.lookupTransform("map", "tracking_camera_45_link_optical", ros::Time(0), _st);
    }    
    catch (tf::TransformException ex) {     
        ROS_ERROR("%s",ex.what());      
    }
    
    /*
    _listener.waitForTransform("tracking_camera_45_link_optical", "map", ros::Time(0), ros::Duration(0.1));
    try {     
        _listener.lookupTransform("tracking_camera_45_link_optical", "map", ros::Time(0), _st);
    }    
    catch (tf::TransformException ex) {     
        ROS_ERROR("%s",ex.what());      
    }
    */

    tf::Quaternion q = _st.getRotation();
    Eigen::Vector4d eigen_q;
    eigen_q << q[3], q[0], q[1], q[2];

    Eigen::Matrix3d Rc_m = QuatToMat( eigen_q );
    Eigen::Vector3d pc_m;
    pc_m << _st.getOrigin().x(), _st.getOrigin().y(), _st.getOrigin().z();
   
    double x = (px - _K_02)/ _K_00;
    double y = (py - _K_12)/ _K_11;

    Eigen::Vector3d rpoint;
    rpoint << x, y, 1.0;
    rpoint = Rc_m * rpoint;
    double lambda_value = (-pc_m[2])/ rpoint[2];

    Eigen::Vector3d down_camera_point;
    down_camera_point << x * lambda_value, y * lambda_value,  lambda_value;

    Eigen::Vector3d map_point;
    map_point = Rc_m*down_camera_point;

    map_point[0] += base_link_pos[0];
    map_point[1] += base_link_pos[1];
    map_point[2] = -map_point[2];
    //map_point[1] = -map_point[1];
    //map_point[2] = -map_point[2];
    cout << "map_point: " << map_point.transpose() << endl;

    //exit(0);
    transform.setOrigin( tf::Vector3(map_point[0], map_point[1], map_point[2]) );  
    transform.setRotation(q0);  
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "target"));    


    geometry_msgs::Point dpoint; 
    dpoint.x = map_point[0];
    dpoint.y = map_point[1];
    dpoint.z = map_point[2];
    _motion_control_destination_pub.publish( dpoint );

    cout << "base_link_pos: " << base_link_pos.transpose() << endl;
    cout << "Rotation matrix: " << endl << Rc_m << endl;
    cout << "pc: " << pc_m.transpose() << endl;
    cout << "Pixel: " << px << " " << py << endl;
    cout << "Point in image space: " << x << " " << y << endl;
    cout << "Rpoint: " << rpoint.transpose() << endl;
    cout << "lambda: " << lambda_value << endl;
    cout << "down_camera_point: " << down_camera_point.transpose() << endl;
    cout << "map_point: " << map_point.transpose() << endl;
    //exit(0);


    return map_point;
}


void SewerDetector::elaborate() {

    ros::Rate r(10);

    while( !_img_ready ) usleep(0.1*1e6);

    vector<Point> outer_ellipse;
    Point ellipse_center;

    while( ros::ok() && !_to_exit ) {
    
        Mat img;
        if( !_src.empty()) {

            if( _to_detect ) {
                img_mutex.lock();
                img = _src;
                img_mutex.unlock();
                get_ellipse( img, outer_ellipse, ellipse_center );
            }
            else {
                usleep(1*1e6);
            }
        }


        if ( _new_destination_point ) {
        
            cout << "Reaching the pixel: " << _destination_point.x << " " << _destination_point.y << endl;
            _new_destination_point = false;
            _to_detect = false;
            calc_destination_point( _destination_point.x, _destination_point.y );
            //calc_destination_point( 640/2, 480/2 );
            //exit(0);

        }

        r.sleep();
    }


    if( _to_exit ) exit(0);
}

void SewerDetector::test_goto() {


}

void SewerDetector::run() {

    if ( TEST_GOTO ) {
        boost::thread test_goto_t(&SewerDetector::test_goto, this);
    }
    else 
        boost::thread elaborate_t(&SewerDetector::elaborate, this);
    ros::spin();
}


int main(int argc, char** argv) {

    ros::init( argc, argv, "sewer_detector");
    
    SewerDetector sd;
    sd.run();

    return 0;

}