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
#include "std_msgs/Int32.h"
#include "geometry_msgs/Pose2D.h"

using namespace std;
using namespace cv; 

std::mutex img_mutex;

#define TEST_GOTO (1)

bool clicked = false;
int clicked_x = 0;
int clicked_y = 0;

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;

    Point seed = Point(x,y);

    clicked_x = x;
    clicked_y = y;
    clicked = true;
}


class DetectorClient {
    public:
        DetectorClient();
        void detection_cb( const sewer_detection::detections & d );
        void img_cb( const sensor_msgs::Image & img );

        void run();
        void client();

    private:
        ros::NodeHandle _nh;
        ros::Publisher  _selected_point_pub;
        ros::Publisher  _seleted_alcantrailla_pub;
        ros::Subscriber _sew_sub;
        ros::Subscriber _img_sub;
        Mat _src;
        bool _img_ready;
        sewer_detection::detections _sewers;

};


DetectorClient::DetectorClient() {


    _selected_point_pub =       _nh.advertise<geometry_msgs::Pose2D>("/detector_client/selected_point", 1);
    _seleted_alcantrailla_pub = _nh.advertise<std_msgs::Int32>("/detector_client/selected_alcantarilla" , 1);

    _sew_sub = _nh.subscribe("/sewer_detection/detections", 1, &DetectorClient::detection_cb, this);
    string image_topic;
    if( !_nh.getParam( "img_tpoic", image_topic)) {
        image_topic = "tracking";
    }
    _img_sub = _nh.subscribe( image_topic, 1, &DetectorClient::img_cb, this);

    _img_ready = false;

}
void DetectorClient::img_cb( const sensor_msgs::Image & img ) {
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


void DetectorClient::detection_cb(const sewer_detection::detections & d) {
    _sewers = d;
}


void DetectorClient::client() {

    ros::Rate r(10);
    namedWindow( "image", 1 );
    setMouseCallback( "image", onMouse, 0 );
    while( !_img_ready ) usleep(0.1*1e6);

    bool done = false;
    Mat img;
    while( ros::ok() && !done ) {
        
        img_mutex.lock();
        _src.copyTo( img );
        img_mutex.unlock();

        if( _sewers.sewers.size() > 0 || TEST_GOTO == 1) {

            if( clicked ) {
                clicked = false;

                if( TEST_GOTO ) {
                    geometry_msgs::Pose2D p;
                    p.x = clicked_x;
                    p.y = clicked_y;
                    _selected_point_pub.publish( p );
                    //todo prevent multiple clicks
                    
                }
                else {
                    cout << "Devo cercare il punto: " << clicked_x << " " << clicked_y << endl;
                

                    int min_index = 1000;
                    int min_dist = 1000;
                    for( int i=0; i<_sewers.sewers.size(); i++ ) {

                        Eigen::Vector2i p; 
                        Eigen::Vector2i s; 
                        p << clicked_x, clicked_y;
                        s << _sewers.sewers[i].point.x,_sewers.sewers[i].point.y;
                        if( min_dist > (p-s).norm() )  {
                            min_dist = (p-s).norm();
                            min_index = _sewers.sewers[i].id;
                        }

                        cout << "Index: " << _sewers.sewers[i].id << endl;
                    

                    }
                }

            }


        }

        imshow( "image", img);
        waitKey(1);
        r.sleep();
    }

}

void DetectorClient::run() {

    boost::thread client_t( &DetectorClient::client, this );
    ros::spin();
}




int main(int argc, char** argv) {

    ros::init( argc, argv, "sewer_detector_client");
    
    DetectorClient dc;
    dc.run();

    return 0;

}