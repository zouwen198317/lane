#ifndef _GarbDetec_FrameProcessor_H_
#define _GarbDetec_FrameProcessor_H_

#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include <opencv2\nonfree\features2d.hpp>

#include <vector>
using std::vector;
using cv::Mat;

enum {
	SURF_FEATURE_DETECTOR = 0,
	SIFT_FEATURE_DETECTOR = 1,

};
//帧处理类，纯虚类
class FrameProcessor 
{
public:
	virtual void initial() = 0;

	// processing method
	virtual void process(cv:: Mat &input, cv:: Mat &output) = 0;


protected:
	int frameCount;
	int frameToStart;
};

//运动目标检测类
class BGFGSegmentor : public FrameProcessor
{
public:
	BGFGSegmentor() : threshold(64), learningRate(0.001) {}

	// processing method
	virtual void process(cv::Mat &frame, cv::Mat &output);

	cv::Mat& getBackgroundImage();
	cv::Mat& getForegroundImage();
	vector<cv::Rect>& getObjRects();

	void setThreshold(int thresh);

protected:
	cv::Mat background; // accumulated background
	cv::Mat foreground; // foreground image
	vector<cv::Rect> objs;
	cv::Mat gray;
	cv::Mat backImage;
	double learningRate; 
	int threshold; // threshold for foreground extraction
	cv::BackgroundSubtractorMOG2 mog;
};

//特征点提取类
class FeatureDetec : public FrameProcessor
{
public:
	FeatureDetec(int detector_type);
	~FeatureDetec();

	virtual void process(cv::Mat &frame, cv::Mat &output);

private:
	cv::Feature2D *pFeatureDetector;
};



#endif