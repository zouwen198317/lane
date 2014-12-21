#include <iostream>
#include <string.h>
#include <fstream>

#include "laneDetection.h"
#include "CKalmanFilter.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int LaneMarkingDetection(string videoName)
{
	VideoCapture cap(videoName);
	laneDetection detect; // object of laneDetection class

	string imname;
	Mat frame;
	Mat img1;// = imread(ippath,0); // Read the image
	cap >> frame;
	cvtColor(frame,img1,CV_BGR2GRAY);
	resize(img1,img1,Size(detect._width,detect._height)); // Resizing the image (only for display purposes)
	
	detect.LMFiltering(img1); // Filtering to detect Lane Markings
 	vector<Vec2f> lines = detect.houghTransform(); // Hough Transform
	Mat imgFinal = detect.drawLines(img1, lines, imname); // draw final Lane Markings on the original image
	imshow("inital frame",imgFinal);
	waitKey(0);

	CKalmanFilter KF2(lines); // Initialization 
	while (cap.isOpened()) {
		//Mat img2 = imread(ippath,0); // Read the image
		Mat img2;
		cap >> frame;
		if (NULL == frame.data){
			break;
		}
		cvtColor(frame,img2,CV_BGR2GRAY);
		resize(img2,img2,Size(detect._width,detect._height)); // Resizing the image (only for display purposes)
		
		detect.LMFiltering(img2); // Filtering to detect Lane Markings
		vector<Vec2f> lines2 = detect.houghTransform(); // Hough Transform
		
		
		// if lanes are not detected, then use the Kalman Filter prediction
		//if (lines2.size() < 2) {
		Mat houghRes = detect.drawLines(img2,lines2, imname); // draw final Lane Markings on the original image for display
		imshow("line detection result",houghRes);
		waitKey(1);
			//oppath += imname;
			//imwrite(oppath,imgFinal); 
			//continue;
		//}
		
		
		///// Kalman Filter to predict the next state
		
		//vector<Vec2f> pp = KF2.predict(); // Prediction

		vector<Vec2f> lines2Final = KF2.update(lines2); // Correction
		//lines = lines2Final; // updating the model
		Mat imgFinal = detect.drawLines(img2,lines2Final, imname); // draw final Lane Markings on the original image
		imshow("kalman filter result",imgFinal);
		waitKey(1);
		/////
		
		//oppath += imname;
	//	imwrite(oppath,imgFinal); // writing the final result to a folder

	}

	return 0;
}