#ifndef	_LANEDETECTION_PROCESSOR_H_
#define _LANEDETECTION_PROCESSOR_H_

#include "FrameProcessor.h"
#include "CKalmanFilter.h"
#include "laneDetection.h"
#include "utils.h"

/********************* Lane Detection ***************************/
class LaneDetecProcessor : public FrameProcessor
{
public:
	LaneDetecProcessor();

	virtual void initial();
	virtual void process(cv::Mat &frame, cv::Mat &output);
	void initialKalman();

public:
	enum {  VERTICAL_EDGES = 0, HORIZONTAL_EDGES = 1, ALL_EDGES = 2 };
	enum { SINGLE_LANE, MULTI_LANE	};

	/* 
	*  detect the edge of a grayscale image gragImage,return the edge image. 
	*/
	Mat findEdges(const Mat& grayImage);

	/*
	*	find line in an edge image edgeImage, return the line paramater (rho, theta).
	*/
	Vec2f findLines(const Mat& edgeImage);

	/*
	* find multi lines in the edge image edgeImage, return the number of lane.
	*/
	int findLines(const Mat& edgeImage, vector<Vec2f>&);

	void getSidePoint(const Vec2f& left, const Vec2f& right);


private:
	int edgeType;
	int laneType;

	float leftSidePos;
	float rightSidePos;
	vector<Mat> frames;
	cv::Point vanishPoint;
	Mat currentFrame;

	CKalmanFilter* m_filter;
	laneDetection* m_detector;
};
#endif