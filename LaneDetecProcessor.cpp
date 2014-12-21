/**
*	Lane Detection
*	date: 2014.12
*/

#include "LaneDetecProcessor.h"
#include "utils.h"

#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

using namespace cv;

#include <iostream>
#include <fstream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;
using std::ofstream;


#define ShowImage

ofstream data_file("line.txt");

enum
{
    LINE_REJECT_DEGREES=60, //in degrees

    CANNY_MIN_THRESHOLD=1,//edge detector mininum hysteresis threshold
    CANNY_MAX_THRESHOLD=100,//edge detector maximum hysteresis threshold

    HOUGH_THRESHOLD=30,     //line approval vote threshold
    HOUGH_MIN_LINE_LENGTH=50, //remove lines shorter than this threshold
    HOUGH_MAX_LINE_GAP=100, //join lines

    LINE_LENGTH_DIFF=10,   //accepted diffenrence of length of lines,

    LANE_DYNAMPARAMS=2,//lane state vector dimension
    LANE_MEASUREPARAMS=2,//lane state vector dimension
    LANE_CONTROLPARAMS=0,//lane state vector dimension

    VEHICLE_DYNAMPARAMS=2,//vehicle state vector dimension
    VEHICLE_MEASUREPARAMS=2,//vehicle measurement dimension
    VEHICLE_CONTROLPARAMS=0,//vehicle control vector

    MAX_LOST_FRAME=30//maximum number of lost frames

};

/************* function declare *******************************/
//detect lines int img, img must be a gray scale image.
static void LineDetec(cv::Mat& img, vector<Line>& lines);
//draw lines to img, img must be color image;
static void DrawLines(cv::Mat& img, vector<Line>& lines);
static void DrawLines(cv::Mat& img, vector<Line>& lines, vector<int>& labels);
static void DrawLines(cv::Mat& img, vector<Vec2f>& lines);
static void DrawLine(cv::Mat& img, Line l);
static void DrawLine(cv::Mat& img, Vec2f, Scalar color = Scalar(0,255,0));
static void LineClusters(const vector<Line>& lines, vector<Line> &clusterLines, vector<int> &labels,int nClusters = 2);
static vector<Point2f> ransac(vector<Point2f> data);
/*
*	根据直线(rho,theta)参数计算其两个端点; 
*	flag = 0 : 表示(rho, theta)是直线的极坐标表示。
*	flag = 1 : 表示(rho, theta)是HoughLines检测直线得到的参数。
*/
static void GetLinePoints(Vec2f, Point2f&, Point2f&, int flag = 0);

/*************** function implementation **********************/
static void DrawLines(cv::Mat& img, vector<Vec2f>& lines)
{
	for (int i = 0; i < lines.size(); ++i) {
		DrawLine(img,lines[i]);
	}
}
static void DrawLine(cv::Mat& img, Vec2f l, Scalar color)
{
	Point2f	p1,p2;
	GetLinePoints(l,p1,p2);
	line(img,p1,p2,color,2);
}

static void LineDetec(const cv::Mat& img, vector<Line>& lines)
{
	lines.clear();
	double rhoThresh = 0.8f, thetaThresh = CV_PI / 90;
	int vote = img.rows / 3;

	const double angle_low = 30.0 / 180 * CV_PI, angle_high = CV_PI - angle_low;
#if 0	/* Probabilistic Hough transform*/
	vector<cv::Vec4i> all_lines;
	HoughLinesP(img,all_lines,rhoThresh,thetaThresh,vote,10,5);

#ifdef ShowImage
	Mat imgColor;
	cvtColor(img,imgColor,CV_GRAY2BGR);
	for( size_t i = 0; i < all_lines.size(); i++ )
	{
		line( imgColor, Point(all_lines[i][0], all_lines[i][1]),Point(all_lines[i][2], all_lines[i][3]), Scalar(0,0,255), 3, 8 );
	}
#endif
	//filter lines
	for (size_t i = 0; i < all_lines.size(); ++i){
		Line l(Point2d(all_lines[i][0],all_lines[i][1]),Point2d(all_lines[i][2],all_lines[i][3]));
#ifdef ShowImage
		Mat singleLineImg;
		cvtColor(img,singleLineImg,CV_GRAY2BGR);
		line(singleLineImg,l.p0,l.p1,Scalar(0,0,255),3);
#endif
		if (l.angle > angle_low && l.angle < angle_high){
			lines.push_back(l);
		}
	}

#else /* traditional Hough transform */
	vector<cv::Vec2f> all_lines;
	HoughLines(img,all_lines,rhoThresh,thetaThresh,vote);
	
	for (size_t i = 0; i < all_lines.size(); ++i) {
		float rho = all_lines[i][0], theta = all_lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		Point2d pt1(cvRound(x0 + 1000*(-b)),cvRound(y0 + 1000*(a)));
		Point2d pt2(cvRound(x0 - 1000*(-b)),cvRound(y0 - 1000*(a)));		
		Line l(pt1,pt2);		
		if (l.angle > angle_low && l.angle < angle_high){
			lines.push_back(l);
		}
#ifdef ShowImage
		Mat singleLineImg;
		cvtColor(img,singleLineImg,CV_GRAY2BGR);
		line(singleLineImg,l.p0,l.p1,Scalar(0,0,255),3);
#endif
	}
#endif /* */

}

static void DrawLines(cv::Mat& img, vector<Line>& lines)
{
	for( size_t i = 0; i < lines.size(); i++ ){
		if ( Point2d(0,0) == lines[i].p0 && Point2d(0,0) == lines[i].p1){
			lines[i].p0.y = 0;
			lines[i].p1.y = img.rows;
			lines[i].p0.x = lines[i].getx(0);
			lines[i].p1.x = lines[i].getx(img.rows);
		}
		line( img, lines[i].p0, lines[i].p1, Scalar(0,255,0), 2, 8 );
	}
}

static void DrawLines(cv::Mat& img, vector<Line>& lines, vector<int> &labels)
{
	Scalar red(0,0,255), blue(255,0,0),yellow(0,200,200);
	Scalar colors[] = {red, blue, yellow };
	for( size_t i = 0; i < lines.size(); i++ ){
		line( img, lines[i].p0, lines[i].p1, colors[labels[i]], 2, 8 );
	}

}
static void DrawLines(cv::Mat& img, Line l)
{
	line( img, l.p0, l.p1, Scalar(0,0,255), 2, 8 );
}

static void GetLinePoints(Vec2f v, Point2f& pt1, Point2f& pt2, int flag)
{
	float r = v[0];
	float t = v[1];
	float x,y;
		
	switch (flag)
	{
	case 0:
		pt2.x = 0;
		pt2.y = cvRound(r / sin(t));
		pt1.y = -pt2.y;
		pt1.x = 2 * cvRound(r / cos(t));
		break;
	case 1:

		x = r*cos(t);
		y = r*sin(t);

		pt1.x = cvRound(x - 1.0*sin(t)*1000);
		pt1.y = y + cos(t)*1000;
		pt2.x = x + 1.0*sin(t)*1000;
		pt2.y = y - cos(t)*1000;

		break;
	default:
		break;

	}
}
static void LineClusters(const vector<Line>& lines, vector<Line> &clusterLines, vector<int> &labels,int nClusters)
{
	int nSamples = lines.size();
	Mat samples(Size(2,nSamples),CV_32F);
	Mat_<float> _samples = samples;
	for (int i = 0; i < nSamples; ++i) {
		_samples(i,0) = lines[i].angle;
		_samples(i,1) = lines[i].dis2origin;
	};

	//归一化数据
	Mat centers;
	if (nClusters > 0) {
		kmeans(samples,nClusters,labels,TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,300,0.001),5,KMEANS_PP_CENTERS,centers);
		clusterLines.clear();


		//int *nLabels = new int[nClusters];
		//float* total_angle = new float[nClusters];
		//float* total_dis = new float[nClusters];
		//for (int i = 0; i < nClusters; ++i) {
		//	total_angle[i] = 0;
		//	total_dis[i] = 0;
		//}
		//for (int i = 0; i < nSamples; ++i) {
		//	nLabels[labels[i]]++;
		//	total_angle[labels[i]] += lines[i].angle;
		//	total_dis[labels[i]] += lines[i].dis2origin;
		//}
		//for (int i = 0 ; i < nClusters; ++i) {
		//	float angle = total_angle[i] / nLabels[i];
		//	float dis = total_dis[i] / nLabels[i];
		//	float k = tan(angle);
		//	float b = dis * sqrt(k*k + 1) * signed(-k);
		//	clusterLines.push_back(Line(k,b));
		//}
	}
}

// Implementing RANSAC to remove outlier lines
// Picking the best estimate having maximum number of inliers
// TO DO: Better implementation 
static vector<Point2f> ransac(vector<Point2f> data){

	float _ransacThres = 0.01;
	
	vector<Point2f> res;
	int maxInliers = 0;

	// Picking up the first sample
	for(int i = 0;i < data.size();i++){
		Point2f p1 = data[i];
	
		// Picking up the second sample
		for(int j = i + 1;j < data.size();j++){
			Point2f p2 = data[j];
			int n = 0;
			
			// Finding the total number of inliers
			for (int k = 0;k < data.size();k++){
				Point2f p3 = data[k];
				float normalLength = norm(p2 - p1);
				float distance = abs((float)((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / normalLength);
				if (distance < _ransacThres) n++;
			}
			
			// if the current selection has more inliers, update the result and maxInliers
			if (n > maxInliers) {
				res.clear();
				maxInliers = n;			
				res.push_back(p1);
				res.push_back(p2);
			}
		
		}
		
	}

	return res;
}
bool sort_line_length(Line l0,Line l1)
{
    return l0.length < l1.length;
}

struct sort_line
{//near vertical line
    bool operator()(Line l0,Line l1)
    {
        return (l0.length*l0.length/fabsf(l0.angle-CV_PI/2) > l1.length*l1.length/fabsf(l1.angle-CV_PI/2));
    }
}sort_line_object;


/*********** class function member *************************/
LaneDetecProcessor::LaneDetecProcessor()
{
	edgeType = VERTICAL_EDGES;
	laneType = SINGLE_LANE;
	frameCount = 0;
	frameToStart = 10;

	m_filter = new CKalmanFilter();
	initialKalman();


}

void LaneDetecProcessor::initialKalman()
{
	m_filter->initial(Vec2f(100,0.8), Vec2f(120,-0.8));
}
void LaneDetecProcessor::initial()
{
}

Mat LaneDetecProcessor::findEdges(const Mat& grayImage)
{
	assert(grayImage.type() == CV_8U);

	/*** edge detection ******/
	Mat edges;
	Mat grad_x,abs_grad_x;
	int thresh = 0;
	edgeType = VERTICAL_EDGES;
	switch (edgeType) 
	{
	case VERTICAL_EDGES:		
		Sobel(grayImage,grad_x,-1,1,0,3);
		convertScaleAbs(grad_x,abs_grad_x);
		threshold(abs_grad_x,edges,40,255,CV_THRESH_BINARY);
		//thresh = threshold(abs_grad_x,edges,0,255,CV_THRESH_OTSU);
		//cout << thresh << endl;
		break;

	case ALL_EDGES:
		Canny(grayImage,edges,40,100);
		break;
	default:
		break;
	}

	//int cannyThreshLow = 40, cannyThreshHigh = 100;
	//Canny(frameRoi,frameCanny,cannyThreshLow,cannyThreshHigh);

	//canny边缘过滤
	//Mat frameDilate,frameCannyFilterd;
	//dilate(frameBinary,frameDilate,getStructuringElement(MORPH_RECT,Size(3,3)));
	//bitwise_and(frameDilate,frameCanny,frameCannyFilterd);
#ifdef ShowImage
	//imshow("abs_grad_x",abs_grad_x);
	imshow("canny frame",edges);
	//imshow("filterd canny",frameCannyFilterd);
	waitKey(1);
#endif

	return edges;
}

Vec2f LaneDetecProcessor::findLines(const Mat& edgeImage)
{
	vector<Line> lines;
	double rhoThresh = 0.8f, thetaThresh = CV_PI / 90;
	int vote = edgeImage.rows / 3;

	const double angle_low = 75.0 / 180 * CV_PI, angle_high = CV_PI - angle_low;
	const double rho_low = 0, rho_high = edgeImage.cols - rho_low;

	/* traditional Hough transform */
	vector<cv::Vec2f> _lines;
	HoughLines(edgeImage,_lines,rhoThresh,thetaThresh,vote);
	vector<Vec2f> retVar;
	
#ifdef ShowImage
	Mat lineImage(edgeImage.size(),CV_8UC3);
	cvtColor(edgeImage,lineImage,CV_GRAY2BGR);
#endif
	/* 滤除角度偏差大的直线 */
	vector<cv::Vec2f>::iterator iter = _lines.begin();
	while (iter != _lines.end()) {
		float rho = (*iter)[0], theta = (*iter)[1];
		if ( (theta > angle_low && theta < angle_high) ||
			 abs(rho) < rho_low || abs(rho) > rho_high )
		{
			iter = _lines.erase(iter);
		}else{
#ifdef ShowImage
			DrawLine(lineImage,*iter);
#endif
			iter++;
			//data_file << rho << " " << theta << endl;
		}
	}
#ifdef ShowImage
	imshow("hough lines",lineImage);
	waitKey(1);
#endif


	if (_lines.size() > 1){
		Mat labels,centers;
		Mat samples = Mat(_lines.size(),2,CV_32F);

		for (int i = 0;i < _lines.size();i++){
			samples.at<float>(i,0) = _lines[i][0];
			samples.at<float>(i,1) = _lines[i][1];
		}
		// K means clustering to get two lines
		kmeans(samples, 1, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );

		// kick out lines far from centers.
		float center_rho = centers.at<float>(0), center_theta = centers.at<float>(1);
		float rho_dis = 50, theta_dis = 20.0 / 180 * CV_PI;
		iter = _lines.begin();
		while (iter != _lines.end()) {
			float rho = (*iter)[0], theta = (*iter)[1];
			if ( abs(rho - center_rho) > rho_dis || abs(theta - center_theta) > theta_dis)
			{
				iter = _lines.erase(iter);
			}else{
				iter++;
			}
		}

		if (_lines.size() == 0) 
			return Vec2f(0,0);
			
		if (_lines.size() > 1) {
			//cluster again to get the center
			samples.create(_lines.size(),2,CV_32F);
			for (int i = 0;i < _lines.size();i++){
				samples.at<float>(i,0) = _lines[i][0];
				samples.at<float>(i,1) = _lines[i][1];
			}
			kmeans(samples, 1, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );
		}
		_lines.clear();
		_lines.push_back(Vec2f(centers.at<float>(0),centers.at<float>(1)));
		

		//vector<Point2f> left;//存放0类直线
		//vector<Point2f> right;//存放1类直线
		//for(int i = 0;i < labels.rows; i++){
		//	if (labels.at<int>(i) == 0) 
		//		left.push_back(Point2f(samples.at<float>(i,0), samples.at<float>(i,1)));
		//	else 
		//		right.push_back(Point2f(samples.at<float>(i,0), samples.at<float>(i,1)));
		//}
		//// Performing Ransac
		//vector<Point2f> leftR = ransac(left); 
		//vector<Point2f> rightR = ransac(right);

		////////////////////

		////if (leftR.size() < 1 || rightR.size() < 1 || 
		//  // (float)(cos((leftR[0].y + leftR[1].y)/2) * cos((rightR[0].y + rightR[1].y)/2)) >= 0) return retVar;
		//if (leftR.size() < 1 || rightR.size() < 1)
		//	return lines;

		//lines.push_back(Line(leftR[0],leftR[1]));
		//lines.push_back(Line(rightR[0],rightR[1]));
		//// //求每类均值，pushing the end points of the line to _lines
		//float leftTotalX(0),leftTotalY(0),rightTotalX(0),rightTotalY(0);
		//for (int i = 0; i < leftR.size(); ++i){
		//	leftTotalX += leftR[i].x;
		//	leftTotalY += leftR[i].y;
		//}
		//for (int j = 0; j < rightR.size(); ++j){
		//	rightTotalX += rightR[j].x;
		//	rightTotalY += rightR[j].y;
		//}
		////先放left线，再放right线
		//if (centers.at<float>(0,1) < centers.at<float>(1,1)) //0类直线theta小，为左边直线
		//{
		//	_lines.push_back(Vec2f(leftTotalX / leftR.size(), leftTotalY / leftR.size()));
		//	_lines.push_back(Vec2f(rightTotalX / rightR.size(), rightTotalY / rightR.size()));
		//}else{			
		//	_lines.push_back(Vec2f(rightTotalX / rightR.size(), rightTotalY / rightR.size()));
		//	_lines.push_back(Vec2f(leftTotalX / leftR.size(), leftTotalY / leftR.size()));
		//}
	}


	//for (int i = 0; i < _lines.size(); ++i) {
	//	float r = _lines[i][0];
	//	float t = _lines[i][1];
	//	
	//	float x = r*cos(t);
	//	float y = r*sin(t);

	//	Point p1(cvRound(x - 1.0*sin(t)*1000), cvRound(y + cos(t)*1000));
	//	Point p2(cvRound(x + 1.0*sin(t)*1000), cvRound(y - cos(t)*1000));
	//	lines.push_back(Line(p1,p2));

	//	//Line l(p1,p2);
	//	//data_file << l.k << " " << l.b << " " << l.angle << " " << l.dis2origin << endl;
	//}
	if (0 == _lines.size())
		_lines.push_back(Vec2f(0,0));

	return _lines[0];
}

int LaneDetecProcessor::findLines(const Mat& edgeImage, vector<Vec2f>& lines)
{
	double rhoPrecision = 0.8f, thetaPrecision = CV_PI / 90;
	int vote = edgeImage.rows / 4;

	//double angle_low = 75.0 / 180 * CV_PI, angle_high = CV_PI - angle_low;
	// double rho_low = 0, rho_high = edgeImage.cols - rho_low;

	/* traditional Hough transform */
	vector<cv::Vec2f> _lines, _leftLines, _rightLines;
	/* 左边rho > 0, 右边 rho < 0;  左边 theta < pi / 2, 右边 theta > pi / 2; */
	HoughLines(edgeImage,_lines,rhoPrecision,thetaPrecision,vote);
	
	/*	
	*	参数转换，将Hough变换得到的直线参数（rho,theta）转换为极坐标参数(rho1,theta1)
	*	rho1 = abs(rho); theta1 = (theta > pi / 2 )? theta - pi : theta;
	*	转换后： 左边直线 0 < theta < pi / 2, 右边直线 -pi / 2 < theta < 0。 
	*	转换后，直线从左过渡到右或者从右过渡到左时，角度就连续了。
	*/
	for (int i = 0; i < _lines.size(); ++i) {
		float rho = fabs(_lines[i][0]), theta = _lines[i][1];
		theta = theta > CV_PI / 2 ?  theta - CV_PI : theta;
		_lines[i][0] = rho;
		_lines[i][1] = theta;
	}
#ifdef ShowImage
	Mat lineImage(edgeImage.size(),CV_8UC3);
	cvtColor(edgeImage,lineImage,CV_GRAY2BGR);
#endif
	
	const float thetaThresh = 10 / 180.0 * CV_PI;
	const float rhoThresh = 50;

	/* find left and right lines and push them to _leftLines and _rightLines. */
	if (m_filter->isInitialing()){ /* kalman filter is initialing */
		for (int i = 0; i < _lines.size(); ++i) {
			float rho = _lines[i][0], theta = _lines[i][1];
			if ( theta > 0) /* belong to left lane */
				_leftLines.push_back(_lines[i]);
			if ( theta <= 0) /* belong to right lane*/
				_rightLines.push_back(_lines[i]);
		}
	}else{ /* kalman filter is steady */
		Vec2f preLeftLine, preRightLine;
		m_filter->getPreState(preLeftLine, preRightLine);
		for (int i = 0; i < _lines.size(); ++i) {
			float rho = _lines[i][0], theta = _lines[i][1];
			if ( fabs(rho - preLeftLine[0]) < rhoThresh &&
				 fabs(theta - preLeftLine[1]) < thetaThresh )
				_leftLines.push_back(_lines[i]);

			if ( fabs(rho - preRightLine[0]) < rhoThresh &&
				 fabs(theta - preRightLine[1]) < thetaThresh )
				_rightLines.push_back(_lines[i]);
		}
	}
#ifdef ShowImage
	imshow("hough lines",lineImage);
	waitKey(1);
#endif

	/* cluster the left lines, use the center to represent the left lane line. */
	_lines.clear();
	/* left lines */
	if (_leftLines.size() >= 1){
		Mat labels,centers;
		Mat samples = Mat(_leftLines.size(),2,CV_32F);

		for (int i = 0;i < _leftLines.size();i++){
			samples.at<float>(i,0) = _leftLines[i][0];
			samples.at<float>(i,1) = _leftLines[i][1];
		}
		// K means clustering to get two lines
		kmeans(samples, 1, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );
		_lines.push_back(Vec2f(centers.at<float>(0),centers.at<float>(1)));
	}else{
		_lines.push_back(Vec2f(0,0));
	} /* if (_leftLine.size() > 1 */

	/* right lines */
	if (_rightLines.size() >= 1){ 
		Mat labels,centers;
		Mat samples = Mat(_rightLines.size(),2,CV_32F);

		for (int i = 0;i < _rightLines.size();i++){
			samples.at<float>(i,0) = _rightLines[i][0];
			samples.at<float>(i,1) = _rightLines[i][1];
		}
		// K means clustering to get two lines
		kmeans(samples, 1, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );
		_lines.push_back(Vec2f(centers.at<float>(0),centers.at<float>(1)));
	}else{
		_lines.push_back(Vec2f(0,0));
	} /* if (_leftLine.size() > 1 */
			
	lines = _lines;
	return _lines.size();
}

void LaneDetecProcessor::process(cv::Mat &frame, cv::Mat &output)
{
	resize(frame,frame,Size(320,240));

	static int imageNumber = 0;

	output.create(frame.size(),frame.type());
	output = frame.clone();

	Mat frameGray;
	if (frame.channels() == 3) {
		cvtColor(frame,frameGray,CV_BGR2GRAY);
	}else{
		frameGray = frame;
	}

#ifdef ShowImage
	line(output,Point(0,output.rows / 2), Point(output.cols, output.rows / 2),Scalar(255,0,255));
	line(output,Point(output.cols / 2,0), Point(output.cols / 2, output.rows),Scalar(255,0,255));
#endif;
	//图像预处理
	Range roiRange = Range(frameGray.rows * 0.6,frameGray.rows);
	Mat frameRoi = frameGray(roiRange,Range::all());
	Mat outputRoi = output(roiRange,Range::all());

	Mat frameBinary, frameEdges;
	frameEdges = this->findEdges(frameRoi);
	
	Rect roiLeft(0,0,frameEdges.cols / 2, frameEdges.rows);
	Rect roiRight(frameEdges.cols / 2, 0 ,frameEdges.cols / 2, frameEdges.rows);
	//Mat edgeLeft = frameEdges(roiLeft);
	//Mat edgeRight = frameEdges(roiRight);
	vector<Vec2f> lines;
	this->findLines(frameEdges, lines);
	

	/*Mat leftLineImageRoi = LineImageRoi(roiLeft);
	Mat rightLineImageRoi = LineImageRoi(roiRight);*/
	/*
	DrawLine(leftLineImageRoi,laneLinesLeft);
	DrawLine(rightLineImageRoi,laneLinesRight);*/
	//DrawLines(leftLineImageRoi,laneLinesLeft);
	//DrawLines(rightLineImageRoi,laneLinesRight);
#ifdef ShowImage
	Mat LineImage = output.clone();
	Mat LineImageRoi = LineImage(roiRange,Range::all());
	DrawLines(LineImageRoi, lines);
	imshow("Line detect result",LineImage);
	waitKey(1);
#endif

	assert(lines.size() == 2);

	Vec2f laneLinesLeft= lines[0], laneLinesRight = lines[1];
	m_filter->update(lines[0],lines[1]);
	if (m_filter->isStoped())
	{
		this->initialKalman();
	}
	if (m_filter->isInitialing())
		cout << "kalman filter is initialing" << endl;
	else
		cout << "kalman filter is stable working " << endl;

	m_filter->getState(laneLinesLeft,laneLinesRight);
	DrawLine(outputRoi,laneLinesLeft,Scalar(0,255,0)); //left lane in green.
	DrawLine(outputRoi,laneLinesRight,Scalar(0,0,255)); // right lane in red.

	m_filter->getPreState(laneLinesLeft,laneLinesRight);
	this->getSidePoint(laneLinesLeft, laneLinesRight);

	/* right line move to middle, the lane is going to change. */
	if (this->rightSidePos < frame.cols / 2) {
		Vec2f tmp = laneLinesLeft;
		laneLinesLeft = laneLinesRight;
		laneLinesRight[0] = tmp[0] + 20;
		laneLinesRight[1] = -tmp[1];
		this->m_filter->changeState(laneLinesLeft, laneLinesRight);
	}

	/* the left line moves to middle, the lane is going to change. */
	if (this->leftSidePos > frame.cols / 2) {
		Vec2f tmp = laneLinesRight;
		laneLinesRight = laneLinesLeft;
		laneLinesLeft[0] = tmp[0]-10;
		laneLinesLeft[1] =  -tmp[1];
		this->m_filter->changeState(laneLinesLeft, laneLinesRight);
	}
	
}

void LaneDetecProcessor::getSidePoint(const Vec2f& left,const Vec2f& right)
{
	Point2f p1,p2;
	GetLinePoints(left,p1,p2);
	Line ll(p1,p2);
	this->leftSidePos = ll.getx(120);

	GetLinePoints(right,p1,p2);
	Line rr(p1,p2);
	this->rightSidePos = rr.getx(120);

}