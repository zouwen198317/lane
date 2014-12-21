#include "FrameProcessor.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <opencv2\nonfree\features2d.hpp>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

#ifdef ShowImage
#include "opencv2/highgui/highgui.hpp"
#endif

/********************* 前景提取 *******************************************/
void BGFGSegmentor::process(cv::Mat &frame, cv::Mat &output) 
{
	//转换为灰度图
	cv::Mat frame_gray(frame.size(),frame.type());
	cv::cvtColor(frame,frame_gray,CV_BGR2GRAY);

	// update the background
	// and return the foreground
	mog(frame,foreground,learningRate);	
	cv::threshold(foreground,foreground,threshold,255,cv::THRESH_BINARY);
	mog.getBackgroundImage(background);
#ifdef ShowImage
	cv::imshow("foreground",foreground);
	cv::imshow("background",background);
	cv::waitKey(10);
#endif
// 	cv::Mat fore = frame - background;
// 	cv::Mat gray(frame.size(),CV_8UC1);
// 	cv::cvtColor(fore,gray,CV_BGR2GRAY);
// 	cv::threshold(gray,gray,threshold,255,cv::THRESH_BINARY);
// 	cv::cvtColor(gray,output,CV_GRAY2BGR);

	//设置ROI
	cv::Rect rect(0,80,640,250);
	cv::Mat imgRoi = foreground(rect);

	//检测目标
	std::vector<std::vector<cv::Point> > rough_contours,contours;
	cv::findContours(imgRoi,rough_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

	//过滤掉长度过小和过大的轮廓；
	const int min_contour_length = 100;
	const int max_contour_length = 1000;
	for (size_t i = 0; i < rough_contours.size(); ++i) {
		if (rough_contours[i].size() > min_contour_length && 
			rough_contours[i].size() < max_contour_length) {
				contours.push_back(rough_contours[i]);
		}
	}

	output.create(frame.size(),CV_8UC3);
	frame.copyTo(output);
	//output = cv::Mat::zeros(frame.size(),CV_8UC3);
	//cv::drawContours(output,contours,-1,cv::Scalar(0,0,255));
	objs.clear();
	for (size_t i = 0; i < contours.size(); ++i) {
		cv::Rect r0= cv::boundingRect(cv::Mat(contours[i]));
		r0.x += rect.x;
		r0.y += rect.y;
		objs.push_back(r0);
		cv::rectangle(output,r0,cv::Scalar(0,0,255),2);
	}
}

cv::Mat& BGFGSegmentor::getBackgroundImage()
{
	mog.getBackgroundImage(background);
	return background;
}

cv::Mat& BGFGSegmentor::getForegroundImage()
{
	return foreground;
}

vector<cv::Rect>& BGFGSegmentor::getObjRects()
{
	return objs;
}
void BGFGSegmentor::setThreshold(int thresh)
{
	threshold = thresh;
}


/************************特征检测 **************************/
FeatureDetec::FeatureDetec(int detector_type)
{
	switch (detector_type)
	{
	//SURF detector
	case SURF_FEATURE_DETECTOR:
		pFeatureDetector = new cv::SurfFeatureDetector(100);
		break;
	//SIFT detector
	case SIFT_FEATURE_DETECTOR:
		pFeatureDetector = new cv::SiftFeatureDetector();
		break;
	default:
		break;
	}
}

FeatureDetec::~FeatureDetec()
{
	delete pFeatureDetector;
	pFeatureDetector = NULL;
}

void FeatureDetec::process(cv::Mat& frame, cv::Mat &output)
{
	vector<cv::KeyPoint> keyPoints;
	pFeatureDetector->detect(frame,keyPoints);
	drawKeypoints(frame,keyPoints,output);
}