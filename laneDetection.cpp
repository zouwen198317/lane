#include <iostream>
#include <string.h>
#include "laneDetection.h"
#include "CKalmanFilter.h"

using namespace cv;

laneDetection::laneDetection(){
	_detectedEdges = Mat();
	_width = 320;
	_height = 240;
	_LMWidth = 10;
	_thres = 40;
	_rho = 1;
	_theta = CV_PI/180.0;
	_houghThres =50;
	_ransacThres = 0.01;
}

laneDetection::~laneDetection(){

}

//////////
// Filter to detect lane Markings
// This is just one approach. Other approaches can be Edge detection, Connected Components, etc.
// The advantage of this approach is that it will only detect the edges in the vertical direction.
void laneDetection::LMFiltering(Mat src){

	_detectedEdges.create(src.size(),CV_8U);
	_detectedEdges.setTo(0);
	Mat frameGray = src;

	//图像预处理
	Range roiRange = Range(frameGray.rows * 0.5,frameGray.rows);
	Mat frameRoi = frameGray(roiRange,Range::all());
	Mat frameBinary;
	int binThresh = 70;
	//threshold(frameRoi,frameBinary,binThresh,255,CV_THRESH_BINARY);
	binThresh = threshold(frameRoi,frameBinary,0,255,CV_THRESH_OTSU);
	threshold(frameRoi,frameBinary,binThresh * 1.1,255,CV_THRESH_BINARY);
/*
	cout << binThresh << endl;*/
	Mat frameCanny = _detectedEdges(roiRange,Range::all());
	int cannyThreshLow = 40, cannyThreshHigh = 100;
	Canny(frameRoi,frameCanny,cannyThreshLow,cannyThreshHigh);

	imshow("detec edges",_detectedEdges);

}
//////////

// Performing Hough Transform
vector<Vec2f> laneDetection::houghTransform(){

	Mat _detectedEdgesRGB;
	cvtColor(_detectedEdges,_detectedEdgesRGB, CV_GRAY2BGR); // converting to RGB
	HoughLines(_detectedEdges,_lines,_rho,_theta,_houghThres); // Finding the hough lines
	vector<Vec2f> retVar;
	
	if (_lines.size() > 1){
		Mat labels,centers;
		Mat samples = Mat(_lines.size(),2,CV_32F);

		for (int i = 0;i < _lines.size();i++){
			samples.at<float>(i,0) = _lines[i][0];
			samples.at<float>(i,1) = _lines[i][1];
		}
		// K means clustering to get two lines
		kmeans(samples, 2, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );

		////////////////// Using RANSAC to get rid of outliers
		_lines.clear();
		
		vector<Point2f> left;//存放0类直线
		vector<Point2f> right;//存放1类直线
		for(int i = 0;i < labels.rows; i++){
			if (labels.at<int>(i) == 0) 
				left.push_back(Point2f(samples.at<float>(i,0), samples.at<float>(i,1)));
			else 
				right.push_back(Point2f(samples.at<float>(i,0), samples.at<float>(i,1)));
		}
		// Performing Ransac
		vector<Point2f> leftR = ransac(left); 
		vector<Point2f> rightR = ransac(right);
		//////////////////
/*
		if (leftR.size() < 1 || rightR.size() < 1 || 
		   (float)(cos((leftR[0].y + leftR[1].y)/2) * cos((rightR[0].y + rightR[1].y)/2)) >= 0) return retVar;*/
		if (leftR.size() < 1 || rightR.size() < 1)
			return retVar;

		
		// //求每类均值，pushing the end points of the line to _lines
		float leftTotalX(0),leftTotalY(0),rightTotalX(0),rightTotalY(0);
		for (int i = 0; i < leftR.size(); ++i){
			leftTotalX += leftR[i].x;
			leftTotalY += leftR[i].y;
		}
		for (int j = 0; j < rightR.size(); ++j){
			rightTotalX += rightR[j].x;
			rightTotalY += rightR[j].y;
		}
		//先放left线，再放right线
		if (centers.at<float>(0,1) < centers.at<float>(1,1)) //0类直线theta小，为左边直线
		{
			_lines.push_back(Vec2f(leftTotalX / leftR.size(), leftTotalY / leftR.size()));
			_lines.push_back(Vec2f(rightTotalX / rightR.size(), rightTotalY / rightR.size()));
		}else{			
			_lines.push_back(Vec2f(rightTotalX / rightR.size(), rightTotalY / rightR.size()));
			_lines.push_back(Vec2f(leftTotalX / leftR.size(), leftTotalY / leftR.size()));
		}
	}


	return _lines;
}

// Implementing RANSAC to remove outlier lines
// Picking the best estimate having maximum number of inliers
// TO DO: Better implementation 
vector<Point2f> laneDetection::ransac(vector<Point2f> data){

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

// Draw Lines on the image
Mat laneDetection::drawLines(Mat img, vector<Vec2f> lines, string name){

	Mat imgRGB;
	cvtColor(img,imgRGB,CV_GRAY2RGB); // converting the image to RGB for display
	vector<Point> endPoints;

	// Here, I convert the polar coordinates to Cartesian coordinates.
	// Then, I extend the line to meet the boundary of the image.
	for (int i = 0;i < lines.size();i++){
		float r = lines[i][0];
		float t = lines[i][1];
		
		float x = r*cos(t);
		float y = r*sin(t);

		Point p1(cvRound(x - 1.0*sin(t)*1000), cvRound(y + cos(t)*1000));
		Point p2(cvRound(x + 1.0*sin(t)*1000), cvRound(y - cos(t)*1000));

		clipLine(img.size(),p1,p2);
		if (p1.y > p2.y){
			endPoints.push_back(p1);
			endPoints.push_back(p2);
		}
		else{
			endPoints.push_back(p2);
			endPoints.push_back(p1);
		}

	}

	///// Finding the intersection point of two lines to plot only lane markings till the intersection
	Point pint;
	bool check = findIntersection(endPoints,pint);

	if (check){
		line(imgRGB,endPoints[0],pint,Scalar(0,0,255),2);//左边直线，red
		line(imgRGB,endPoints[2],pint,Scalar(0,255,0),2);//右边直线,green
	}	
	/////

	// Saving to intercepts.csv
//	float xIntercept = min(endPoints[0].x,endPoints[2].x);
	//myfile << name << "," << xIntercept * 2 << "," << pint.x * 2 << endl;

	//visualize(imgRGB); // Visualize the final result

	return imgRGB;
}

// Finding the Vanishing Point
bool laneDetection::findIntersection(vector<Point> endP, Point& pi){
	
	if (0 == endP.size() || endP.size()!= 4)
		return false;

	Point x = endP[2] - endP[0];
	Point d1 = endP[1] - endP[0];
	Point d2 = endP[3] - endP[2];
	
	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < 1e-8) // No intersection
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    pi = endP[0] + d1 * t1;
    return true;


}

// Visualize
void laneDetection::visualize(Mat imgRGB){
	
	namedWindow("LaneMarkings");
	imshow("LaneMarkings",imgRGB);
	waitKey(1);

}

#ifdef LaneTest
int main()
{
	laneDetection detect; // object of laneDetection class

	string ippath = "./images/";
	string oppath = "./output/";
	string imname;
	ifstream imageNames ("imNames.txt");
	getline(imageNames,imname);
    
	ippath += imname;
	Mat img1 = imread(ippath,0); // Read the image
	resize(img1,img1,Size(detect._width,detect._height));
	
	detect.LMFiltering(img1); // Filtering to detect Lane Markings
	vector<Vec2f> lines = detect.houghTransform(); // Hough Transform
	Mat imgFinal = detect.drawLines(img1, lines, imname); // draw final Lane Markings on the original image for display
	
	oppath += imname;
	imwrite(oppath,imgFinal); 

	while ( getline (imageNames,imname) ){
		ippath = "./images/";
		oppath = "./output/";
		ippath += imname;

		Mat img2 = imread(ippath,0); // Read the image
		resize(img2,img2,Size(detect._width,detect._height));
		
		detect.LMFiltering(img2); // Filtering to detect Lane Markings
		vector<Vec2f> lines2 = detect.houghTransform(); // Hough Transform
		
		
		// if lanes are not detected, then use the Kalman Filter prediction
		if (lines2.size() < 2) {
			imgFinal = detect.drawLines(img2,lines, imname); // draw final Lane Markings on the original image for display
			oppath += imname;
			imwrite(oppath,imgFinal); 
			continue;
		}
		
		///// Kalman Filter to predict the next state
		CKalmanFilter KF2(lines);
		vector<Vec2f> pp = KF2.predict();

		vector<Vec2f> lines2Final = KF2.update(lines2);
		lines = lines2Final;
		imgFinal = detect.drawLines(img2,lines2, imname); // draw final Lane Markings on the original image for display
		/////
		
		oppath += imname;
		imwrite(oppath,imgFinal);
	}
	

}
#endif