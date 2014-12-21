#include <iostream>
#include "CKalmanFilter.h"

using namespace cv;
using namespace std;


// Constructor
CKalmanFilter::CKalmanFilter()
{

	_isInitialing = true;
	acceptNumber = 0;
	rejectNumber = 0;
		/*
	*	4 measurement and 8 state parameters
	*	states : rho1,theta1,rho2,theta2,delta_rho1,delta_theta1,delta_rho2,delta_theta2;
	*/
	kalman = new KalmanFilter( 8, 4, 0 ); 
	kalman->transitionMatrix = (Mat_<float>(8, 8) << 1,0,0,0, 1,0,0,0, 
													 0,1,0,0, 0,1,0,0,
													 0,0,1,0, 0,0,1,0,
													 0,0,0,1, 0,0,0,1,
													 0,0,0,0, 1,0,0,0,
													 0,0,0,0, 0,1,0,0,
													 0,0,0,0, 0,0,1,0,
													 0,0,0,0, 0,0,0,1);

	// Initialization
	kalman->statePost.setTo(0);

	setIdentity(kalman->measurementMatrix);
	setIdentity(kalman->processNoiseCov, Scalar::all(1e-4));
	setIdentity(kalman->measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kalman->errorCovPost, Scalar::all(1));
}

CKalmanFilter::CKalmanFilter(vector<Vec2f> p){
	/*
	*	4 measurement and 8 state parameters
	*	states : rho1,theta1,rho2,theta2,delta_rho1,delta_theta1,delta_rho2,delta_theta2;
	*/
	kalman = new KalmanFilter( 8, 4, 0 ); 
	kalman->transitionMatrix = (Mat_<float>(8, 8) << 1,0,0,0, 1,0,0,0, 
													 0,1,0,0, 0,1,0,0,
													 0,0,1,0, 0,0,1,0,
													 0,0,0,1, 0,0,0,1,
													 0,0,0,0, 1,0,0,0,
													 0,0,0,0, 0,1,0,0,
													 0,0,0,0, 0,0,1,0,
													 0,0,0,0, 0,0,0,1);

	// Initialization
	//kalman->statePre.at<float>(0) = p[0][0]; // r1
	//kalman->statePre.at<float>(1) = p[0][1]; // theta1
	//kalman->statePre.at<float>(2) = p[1][0]; // r2
	//kalman->statePre.at<float>(3) = p[1][1]; // theta2

	kalman->statePost.at<float>(0)=p[0][0];
	kalman->statePost.at<float>(1)=p[0][1];
	kalman->statePost.at<float>(2)=p[1][0];
	kalman->statePost.at<float>(3)=p[1][1];

	setIdentity(kalman->measurementMatrix);
	setIdentity(kalman->processNoiseCov, Scalar::all(1e-4));
	setIdentity(kalman->measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kalman->errorCovPost, Scalar::all(10));
}

// Destructor
CKalmanFilter::~CKalmanFilter(){
	delete kalman;
}

void CKalmanFilter::initial(const Vec2f& left, const Vec2f& right)
{
	_isInitialing = true;
	acceptNumber = 0;
	rejectNumber = 0;
		/*
	*	4 measurement and 8 state parameters
	*	states : rho1,theta1,rho2,theta2,delta_rho1,delta_theta1,delta_rho2,delta_theta2;
	*/
	kalman->init( 8, 4, 0 ); 
	kalman->transitionMatrix = (Mat_<float>(8, 8) << 1,0,0,0, 1,0,0,0, 
													 0,1,0,0, 0,1,0,0,
													 0,0,1,0, 0,0,1,0,
													 0,0,0,1, 0,0,0,1,
													 0,0,0,0, 1,0,0,0,
													 0,0,0,0, 0,1,0,0,
													 0,0,0,0, 0,0,1,0,
													 0,0,0,0, 0,0,0,1);

	// Initialization

	kalman->statePost.at<float>(0)=left[0];
	kalman->statePost.at<float>(1)=left[1];
	kalman->statePost.at<float>(2)=right[0];
	kalman->statePost.at<float>(3)=right[1];

	kalman->statePre.at<float>(0)=left[0];
	kalman->statePre.at<float>(1)=left[1];
	kalman->statePre.at<float>(2)=right[0];
	kalman->statePre.at<float>(3)=right[1];

	setIdentity(kalman->measurementMatrix);
	setIdentity(kalman->processNoiseCov, Scalar::all(1e-4));
	setIdentity(kalman->measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kalman->errorCovPost, Scalar::all(1));

}

void CKalmanFilter::changeState(const Vec2f& left, const Vec2f& right)
{
	this->initial(left,right);
	this->_isInitialing = false;
}
vector<Vec2f> CKalmanFilter::update(vector<Vec2f>)
{
	vector<Vec2f> lines;

	return lines;
}

bool CKalmanFilter::isInitialing()
{
	return _isInitialing;
}

bool CKalmanFilter::isStoped()
{
	return rejectNumber > REJECT_THRESH;
}

// Correct the prediction based on the measurement
void CKalmanFilter::update(const Vec2f& left,const Vec2f& right){

	if (Vec2f(0,0) == left || Vec2f(0,0) == right) {
		rejectNumber++;
		return;
	}

	Mat_<float> measurement(4,1);
	measurement.setTo(Scalar(0));

	measurement.at<float>(0) = left[0];
	measurement.at<float>(1) = left[1];
	measurement.at<float>(2) = right[0];
	measurement.at<float>(3) = right[1];

	if ( Vec2f(0,0) == left ){
		measurement.at<float>(0) = kalman->statePre.at<float>(0);
		measurement.at<float>(1) = kalman->statePre.at<float>(1);
	}
	if ( Vec2f(0,0) == right) {
		measurement.at<float>(2) = kalman->statePre.at<float>(2);
		measurement.at<float>(3) = kalman->statePre.at<float>(3);
	}
	 
	float theta_thresh, rho_thresh = 50.0f;
	if (_isInitialing) {
		theta_thresh = 60 / 180.0 * CV_PI;
		rho_thresh = 150;
	}else {
		theta_thresh = 10 / 180.0 * CV_PI;
		rho_thresh = 50.0f;
	}

	Mat estimated;
	Mat state = kalman->measurementMatrix * kalman->statePost;
	Mat diff = abs(measurement - state);
	Mat_<float> _diff = diff;
	//cout << diff << endl;
	if (_diff(0) < rho_thresh && _diff(1) < theta_thresh &&
		_diff(2) < rho_thresh && _diff(3) < theta_thresh) 
	{
		kalman->predict();
		kalman->correct(measurement);
		this->rejectNumber = 0;
		if (++acceptNumber > ACCEPT_THRESH)
			_isInitialing = false;
	}else {
		this->rejectNumber++;
	}

	waitKey(1);
	
	return ; // return the measurement

}

void CKalmanFilter::getState(Vec2f& left, Vec2f& right)
{
	vector<Vec2f> lines;
	float rho1 = kalman->statePost.at<float>(0);
	float theta1 = kalman->statePost.at<float>(1);
	float rho2 = kalman->statePost.at<float>(2);
	float theta2 = kalman->statePost.at<float>(3);

	left = Vec2f(rho1,theta1);
	right = Vec2f(rho2,theta2);
}

void CKalmanFilter::getPreState(Vec2f& left, Vec2f& right)
{
	vector<Vec2f> lines;
	float rho1 = kalman->statePre.at<float>(0);
	float theta1 = kalman->statePre.at<float>(1);
	float rho2 = kalman->statePre.at<float>(2);
	float theta2 = kalman->statePre.at<float>(3);

	left = Vec2f(rho1,theta1);
	right = Vec2f(rho2,theta2);
}

vector<Line> CKalmanFilter::update(vector<Line>& lines)
{
	if (lines.size() != 2) {
		lines.clear();
		lines.push_back(Line(0,0));
		lines.push_back(Line(0,0));
	}
	Mat_<float> measurement(4,1);
	measurement.setTo(Scalar(0));

	measurement.at<float>(0) = lines[0].angle;
	measurement.at<float>(1) = lines[0].dis2origin;
	measurement.at<float>(2) = lines[1].angle;
	measurement.at<float>(3) = lines[1].dis2origin;

	float angle_thresh = 10000, dis_thresh = 500000.0f;

	Mat estimated;
	estimated = kalman->predict();

	Mat state = kalman->measurementMatrix * kalman->statePost;
	Mat diff = abs(measurement - state);
	Mat_<float> _diff = diff;
	//cout << diff << endl;
	if (_diff(0) < angle_thresh && _diff(1) < dis_thresh &&
		_diff(2) < angle_thresh && _diff(3) < dis_thresh) 
	{
		kalman->correct(measurement);
	}

	// Update the measurement	
	Mat_<float> _estmated = estimated;
	lines.clear();
	lines.push_back(Line(_estmated(0),_estmated(1)));
	lines.push_back(Line(_estmated(2),_estmated(3)));

	waitKey(1);
	
	return lines; // return the measurement
}