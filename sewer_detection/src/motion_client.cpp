#include "ros/ros.h"
#include "boost/thread.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TwistStamped.h"
#include "mavros_msgs/State.h"
#include "mavros_msgs/CommandBool.h"
#include "mavros_msgs/CommandTOL.h"
#include "mavros_msgs/PositionTarget.h"
#include "Eigen/Dense"
#include "sensor_msgs/Joy.h"
#include "sensor_msgs/Imu.h"

using namespace Eigen;
using namespace std;

inline Eigen::Vector3d R2XYZ(Eigen::Matrix3d R) {
    double phi=0.0, theta=0.0, psi=0.0;
    Vector3d XYZ = Vector3d::Zero();
    
    theta = asin(R(0,2));
    
    if(fabsf(cos(theta))>pow(10.0,-10.0))
    {
        phi=atan2(-R(1,2)/cos(theta), R(2,2)/cos(theta));
        psi=atan2(-R(0,1)/cos(theta), R(0,0)/cos(theta));
    }
    else
    {
        if(fabsf(theta-M_PI/2.0)<pow(10.0,-5.0))
        {
            psi = 0.0;
            phi = atan2(R(1,0), R(2,0));
            theta = M_PI/2.0;
        }
        else
        {
            psi = 0.0;
            phi = atan2(-R(1,0), R(2,0));
            theta = -M_PI/2.0;
        }
    }
    
    XYZ << phi,theta,psi;
    return XYZ;
}

inline Matrix3d QuatToMat(Vector4d Quat){
    Matrix3d Rot;
    float s = Quat[0];
    float x = Quat[1];
    float y = Quat[2];
    float z = Quat[3];
    Rot << 1-2*(y*y+z*z),2*(x*y-s*z),2*(x*z+s*y),
    2*(x*y+s*z),1-2*(x*x+z*z),2*(y*z-s*x),
    2*(x*z-s*y),2*(y*z+s*x),1-2*(x*x+y*y);
    return Rot;
}

class MotionClient {

    public:
        MotionClient();
        void position_controller();
		void run();
		void localization_cb ( geometry_msgs::PoseStampedConstPtr msg );
		void mavros_state_cb( mavros_msgs::State mstate);
        void joy_cb( sensor_msgs::JoyConstPtr j );
        void joy_ctrl();
        void main_loop();
        void gen_traj( geometry_msgs::Point dp, double cv );
        void get_target_point_cb( geometry_msgs::Point p);
        
    private:

        ros::NodeHandle _nh;
        ros::Publisher _target_pub;

        ros::Subscriber _localization_sub;
        ros::Subscriber _joy_data_sub;
        ros::Subscriber _mavros_state_sub;
        ros::Subscriber _target_point_sub;

        bool _first_local_pos;
        bool _enable_joy;
        bool _joy_ctrl;
        bool _joy_ctrl_active;

        Vector3d _cmd_p;
        double _cmd_yaw;       
        Vector3d _w_p;
        Vector3d _vel_joy;
        Vector4d _w_q;
        float _mes_yaw;
        mavros_msgs::State _mstate;
        double _vel_joy_dyaw;

        geometry_msgs::Point _dp;
        bool _new_trajectory_tp;
};



MotionClient::MotionClient() {

	_target_pub = _nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);

    _localization_sub   = _nh.subscribe( "/mavros/local_position/pose", 1, &MotionClient::localization_cb, this);
    _mavros_state_sub   = _nh.subscribe( "/mavros/state", 1, &MotionClient::mavros_state_cb, this);
    _joy_data_sub       = _nh.subscribe("/joy", 1, &MotionClient::joy_cb, this);
    _target_point_sub   = _nh.subscribe("/motion_client/target_point", 1, &MotionClient::get_target_point_cb, this);

    _vel_joy << 0.0, 0.0, 0.0;

    _first_local_pos = false;
    _enable_joy = false;

    _joy_ctrl_active =  false;    
    _new_trajectory_tp = false;
}

void MotionClient::mavros_state_cb( mavros_msgs::State mstate) {
    _mstate = mstate;
}


//--------------------------5 order--------------------------------------------
// AX = B
// A = inv(X)*B - here we compute matrix A
Eigen::MatrixXd computeQuinticCoeff(double t0, double tf, std::vector<double> vec_q0, std::vector<double> vec_qf)
{

    Eigen::MatrixXd X(6, 6);
    Eigen::MatrixXd B(6, 1);

    X(0, 0) = 1;
    X(0, 1) = t0;
    X(0, 2) = std::pow(t0, 2);
    X(0, 3) = std::pow(t0, 3);
    X(0, 4) = std::pow(t0, 4);
    X(0, 5) = std::pow(t0, 5);

    X(1, 0) = 0;
    X(1, 1) = 1;
    X(1, 2) = 2 * t0;
    X(1, 3) = 3 * std::pow(t0, 2);
    X(1, 4) = 4 * std::pow(t0, 3);
    X(1, 5) = 5 * std::pow(t0, 4);

    X(2, 0) = 0;
    X(2, 1) = 0;
    X(2, 2) = 2;
    X(2, 3) = 6 * t0;
    X(2, 4) = 12 * std::pow(t0, 2);
    X(2, 5) = 20 * std::pow(t0, 3);

    X(3, 0) = 1;
    X(3, 1) = tf;
    X(3, 2) = std::pow(tf, 2);
    X(3, 3) = std::pow(tf, 3);
    X(3, 4) = std::pow(tf, 4);
    X(3, 5) = std::pow(tf, 5);

    X(4, 0) = 0;
    X(4, 1) = 1;
    X(4, 2) = 2 * tf;
    X(4, 3) = 3 * std::pow(tf, 2);
    X(4, 4) = 4 * std::pow(tf, 3);
    X(4, 5) = 5 * std::pow(tf, 4);

    X(5, 0) = 0;
    X(5, 1) = 0;
    X(5, 2) = 2;
    X(5, 3) = 6 * tf;
    X(5, 4) = 12 * std::pow(tf, 2);
    X(5, 5) = 20 * std::pow(tf, 3);

    B(0, 0) = vec_q0[0];
    B(1, 0) = vec_q0[1];
    B(2, 0) = vec_q0[2];
    B(3, 0) = vec_qf[0];
    B(4, 0) = vec_qf[1];
    B(5, 0) = vec_qf[2];

    return (X.inverse() * B);
    
}

//-----------------------5 order-----------------------------------------------
void computeQuinticTraj(Eigen::MatrixXd A, double t0, double tf, int n, std::vector<double> & qd, std::vector<double> & d_qd, std::vector<double> & dd_qd, std::vector<double> & time) {
    std::vector<double> a = {A(0, 0), A(1, 0), A(2, 0), A(3, 0), A(4, 0), A(5, 0)};

    float step = (tf - t0) / n;
    for (float t = t0; t < tf; t += step) {

        float qdi = a[0] + a[1] * t + a[2] * std::pow(t, 2) + a[3] * std::pow(t, 3) + a[4] * std::pow(t, 4) + a[5] * std::pow(t, 5);
        float d_qdi = a[1] + 2 * a[2] * t + 3 * a[3] * std::pow(t, 2) + 4 * a[4] * std::pow(t, 3) + 5 * a[5] * std::pow(t, 4);
        float dd_qdi = 2 * a[2] + 6 * a[3] * t + 12 * a[4] * std::pow(t, 2) + 20 * a[5] * std::pow(t, 3);

        qd.push_back(qdi);
        d_qd.push_back(d_qdi);
        dd_qd.push_back(dd_qdi);
        time.push_back(t);
    }    
}


void MotionClient::gen_traj( geometry_msgs::Point dp, double cv ) {

    cout << "Generate trajectory towards: " << dp.x << " " << dp.y << " " << dp.z << endl;
    //return;
    double vel_0, acc_0, vel_f, acc_f;
    std::vector < std::vector < double> > p_traj;

    p_traj.resize(3);    

    vel_0 = acc_0 = 0.0;
    vel_f = acc_f = 0.0;    
    double x_0;
    double y_0;
    double z_0;
    x_0 = _cmd_p[0];
    y_0 = _cmd_p[1];
    z_0 = _cmd_p[2]; //Current position

    double x_f = dp.x;
    double y_f = dp.y;
    double z_f = dp.z;
    
    std::vector<double> vec_x0{x_0, vel_0, acc_0};
    std::vector<double> vec_xf{x_f, vel_f, acc_f};

    std::vector<double> vec_y0{y_0, vel_0, acc_0};
    std::vector<double> vec_yf{y_f, vel_f, acc_f};

    std::vector<double> vec_z0{z_0, vel_0, acc_0};
    std::vector<double> vec_zf{z_f, vel_f, acc_f};

    double dt = 0.01;
    double t0 = 0.0;

    Eigen::Vector3d pi;
    Eigen::Vector3d pf;

    pi << x_0, y_0, z_0;
    pf << x_f, y_f, z_f;

    double tf = (pf-pi).norm() / cv;
    double n_points = tf * 1/dt;
    
    int np = ceil( n_points );
    
    Eigen::MatrixXd traj_x_A = computeQuinticCoeff(t0, tf, vec_x0, vec_xf);    
    Eigen::MatrixXd traj_y_A = computeQuinticCoeff(t0, tf, vec_y0, vec_yf);
    Eigen::MatrixXd traj_z_A = computeQuinticCoeff(t0, tf, vec_z0, vec_zf);


    std::vector<double> qd_x, qd_y, qd_z;
    std::vector<double> d_qd_x, d_qd_y, d_qd_z;
    std::vector<double> dd_qd_x, dd_qd_y, dd_qd_z;
    std::vector<double> time_x, time_y, time_z;
    computeQuinticTraj(traj_x_A, t0, tf,   np,   qd_x, d_qd_x, dd_qd_x, time_x);
    computeQuinticTraj(traj_y_A, t0, tf,   np,   qd_y, d_qd_y, dd_qd_y, time_y);
    computeQuinticTraj(traj_z_A, t0, tf,   np,   qd_z, d_qd_z, dd_qd_z, time_z);

    for(int pp=0; pp<qd_x.size(); pp++ ) {
        p_traj[0].push_back( qd_x[pp] );
        p_traj[1].push_back( qd_y[pp] );
        p_traj[2].push_back( qd_z[pp] );
    }
    
    
    
    ros::Rate r(100);
    int j=0;
    
    while (j < p_traj[0].size() ) {
    
        _cmd_p[0] = p_traj[0][j];
        _cmd_p[1] = p_traj[1][j];
        _cmd_p[2] = p_traj[2][j];        
        j++;

        
        r.sleep();      
    }
    
    cout << "Traj point: " << p_traj[0][p_traj[0].size()-1] << " " << p_traj[1][p_traj[0].size()-1] << " " << p_traj[2][p_traj[0].size()-1] << endl;
}

void MotionClient::localization_cb ( geometry_msgs::PoseStampedConstPtr msg ) {
    _w_p << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    
    
    Eigen::Vector3d rpy = R2XYZ ( QuatToMat ( Eigen::Vector4d( msg->pose.orientation.w,  msg->pose.orientation.x,  msg->pose.orientation.y,  msg->pose.orientation.z) ) );
    _mes_yaw = rpy(2);

    Quaternionf q;
    q = AngleAxisf(0.0, Vector3f::UnitX())
        * AngleAxisf(0.0, Vector3f::UnitY())
        * AngleAxisf(_mes_yaw, Vector3f::UnitZ());
    Vector4d w_q ( q.w(), q.x(), q.y(), q.z() );
    _w_q = w_q / w_q.norm() ;
    

    _first_local_pos = true;    
}

void MotionClient::position_controller(){

    ros::Rate r(100);

    mavros_msgs::PositionTarget ptarget;
    ptarget.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    ptarget.type_mask =
    mavros_msgs::PositionTarget::IGNORE_VX |
    mavros_msgs::PositionTarget::IGNORE_VY |
    mavros_msgs::PositionTarget::IGNORE_VZ |
    mavros_msgs::PositionTarget::IGNORE_AFX |
    mavros_msgs::PositionTarget::IGNORE_AFY |
    mavros_msgs::PositionTarget::IGNORE_AFZ |
    mavros_msgs::PositionTarget::FORCE;

    while( !_first_local_pos )
        usleep(0.1*1e6);
    ROS_INFO("First local pose arrived!");

    _cmd_p = _w_p;
    _cmd_yaw = _mes_yaw;

    while (ros::ok()) {
        if( _mstate.mode != "OFFBOARD" ) {
            _cmd_p = _w_p;
            _cmd_yaw = _mes_yaw;
        } // No control: follow localization

        //---Publish command
        ptarget.header.stamp = ros::Time::now();
        ptarget.position.x = _cmd_p[0];
        ptarget.position.y = _cmd_p[1];
        ptarget.position.z = _cmd_p[2];
        ptarget.yaw = _cmd_yaw;
        //cout << "cmd p: " << _cmd_p.transpose() << endl;
        _target_pub.publish( ptarget );
        //---

        r.sleep();
    }
}


void MotionClient::joy_cb( sensor_msgs::JoyConstPtr j ) {  
    _vel_joy[0] = j->axes[1]*0.2;
    _vel_joy[1] = j->axes[0]*0.2;
    _vel_joy[2] = j->axes[4]*0.2;
    _vel_joy_dyaw = j->axes[3]*0.2;


    if( j->buttons[0] == 1 ) _enable_joy = true;
    else _enable_joy = false;

}

void MotionClient::joy_ctrl () {

    ros::Rate r(100);
    
    _joy_ctrl_active = true;
    cout << "Activating joy control" << endl;
    
    while ( ros::ok() && _joy_ctrl ) {

        if( _mstate.mode == "OFFBOARD" ) {
            _cmd_p[0] += _vel_joy[0]*(1/100.0);
            _cmd_p[1] += _vel_joy[1]*(1/100.0);
            _cmd_p[2] += _vel_joy[2]*(1/100.0);
            _cmd_yaw += _vel_joy_dyaw*(1/100.0);
        }

        _joy_ctrl_active = true;
        r.sleep();
    }

    _joy_ctrl_active = false; 
}


void MotionClient::get_target_point_cb( geometry_msgs::Point p) {

    _new_trajectory_tp = true;
    _dp = p;
    


}



void MotionClient::main_loop () {

    int enable_joy_cnt = 0;

    _joy_ctrl_active = false;
    _joy_ctrl = false;
    
    ros::Rate r(10);

    enable_joy_cnt = 0;

    while( ros::ok() ) {

        enable_joy_cnt++;
        
        if( _enable_joy == true && enable_joy_cnt > 50) {
            _joy_ctrl = !_joy_ctrl;
            enable_joy_cnt = 0;
            _enable_joy = false;
        }

        //Enable disable joy ctrl
        if( _joy_ctrl && !_joy_ctrl_active ) {     
            boost::thread joy_ctrl_t (&MotionClient::joy_ctrl, this);
            usleep(0.2*1e6);
        }
        else if (!_joy_ctrl ) {
            if (_joy_ctrl_active == true ) {
                _joy_ctrl_active = false;
            }
        }

        if( _new_trajectory_tp ) {
            _joy_ctrl = false;
            boost::thread traj_t(&MotionClient::gen_traj, this, _dp, 0.3);
            _new_trajectory_tp = false;
        }

        r.sleep();
    }
}

void MotionClient::run(){
    boost::thread position_controller_t( &MotionClient::position_controller, this);
    boost::thread main_loop_t( &MotionClient::main_loop, this );
    ros::spin();
}


int main(int argc, char** argv ) {
    ros::init(argc, argv, "motion_client");
    MotionClient sc;
    sc.run();
    return 0;
}