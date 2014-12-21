#include "opencv2/video/video.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "VideoProcessor.h"
#include "LaneDetecProcessor.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

#define ShowImage


extern int detectLane(cv::Mat &frame,std::vector<cv::Vec4i>lines);
extern int trackline(cv::Mat &frame,std::vector<cv::Vec4i>& lines);

extern int SimpleLaneTrackingMaster(string videoName);
extern int LaneMarkingDetection(string videoName);
extern int KalmanTest();

void DrawLine(Mat& img, Vec2f v,Scalar color = Scalar(0,255,0)){

	float r = v[0], t = v[1];
	Point pt1,pt2;
	pt2.x = 0;
	pt2.y = cvRound(r / sin(t));
	pt1.y = -pt2.y;
	pt1.x = 2 * cvRound(r / cos(t));
	line(img,pt1,pt2,color);
}
void ModelingPrepare()
{
	int fileNumber = 100;
	LaneDetecProcessor processor;
	float rho,theta;
	ofstream file("../lane_image/GP019932.txt");
	int imageNumber(0);
	for (int i = 0; i < fileNumber; ++i) {
		Mat src = imread(format("../lane_image/GP019933/%d.jpg",i),0);
		if (!src.data)
			break;
		cout << imageNumber++ << endl;
		Mat img = src(Range(src.rows*0.6,src.rows),Range::all());
		Mat edgeImage = processor.findEdges(img);

		double rhoPrecision = 0.5f, thetaPrecision = CV_PI / 180;
		int vote = 25;

		//double angle_low = 75.0 / 180 * CV_PI, angle_high = CV_PI - angle_low;
		// double rho_low = 0, rho_high = edgeImage.cols - rho_low;

		/* traditional Hough transform */
		vector<cv::Vec2f> _lines;
		/* 左边rho > 0, 右边 rho < 0;  左边 theta < pi / 2, 右边 theta > pi / 2; */
		HoughLines(edgeImage,_lines,rhoPrecision,thetaPrecision,vote);
		Mat houghImage;
		cvtColor(img,houghImage,CV_GRAY2BGR);
		for (int i = 0; i < _lines.size(); ++i) {
			DrawLine(houghImage,_lines[i]);
		}
		imshow("hough lines",houghImage);
		waitKey(1);

		/*	
		*	参数转换，将Hough变换得到的直线参数（rho,theta）转换为极坐标参数(rho1,theta1)
		*	rho1 = abs(rho); theta1 = (theta > pi / 2 )? theta - pi : theta;
		*	转换后： 左边直线 0 < theta < pi / 2, 右边直线 -pi / 2 < theta < 0。 
		*	转换后，直线从左过渡到右或者从右过渡到左时，角度就连续了。
		*/
		float thetaThresh = 60 / 180.0 * CV_PI; 
		vector<cv::Vec2f>::iterator iter = _lines.begin();
 		while( iter!= _lines.end()) {
			float rho = fabs((*iter)[0]), theta = (*iter)[1];
			theta = theta > CV_PI / 2 ?  theta - CV_PI : theta;
			if (theta > thetaThresh || theta < -thetaThresh){
				iter = _lines.erase(iter);
			}else{
				(*iter)[0] = rho;
				(*iter)[1] = theta;
				iter++;
			}
		}

		Mat colorImage;
		cvtColor(img,colorImage,CV_GRAY2BGR);
		for (int i = 0; i < _lines.size(); ++i) {
			DrawLine(colorImage,_lines[i]);
		}
		imshow("filtered hough lines",colorImage);


		/* cluster the left lines, use the center to represent the left lane line. */
		/* left lines */
		if (_lines.size() >= 2){
			Mat labels,centers;
			Mat samples = Mat(_lines.size(),2,CV_32F);

			for (int i = 0;i < _lines.size();i++){
				samples.at<float>(i,0) = _lines[i][0];
				samples.at<float>(i,1) = _lines[i][1];
			}
			// K means clustering to get two lines
			kmeans(samples, 2, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );

			_lines.clear();
			_lines.push_back(Vec2f(centers.at<float>(0,0),centers.at<float>(0,1)));
			_lines.push_back(Vec2f(centers.at<float>(1,0),centers.at<float>(1,1)));
		}


		Mat clusterImage;
		cvtColor(img,clusterImage,CV_GRAY2BGR);
		for (int i = 0; i < _lines.size(); ++i) {
			DrawLine(clusterImage,_lines[i],Scalar(0,0,255));
		}

		imshow("houghlines",clusterImage);
		char c = waitKey(0);
		if (c != 's')
			continue;
		/*	
		*	参数转换，将Hough变换得到的直线参数（rho,theta）转换为极坐标参数(rho1,theta1)
		*	rho1 = abs(rho); theta1 = (theta > pi / 2 )? theta - pi : theta;
		*	转换后： 左边直线 0 < theta < pi / 2, 右边直线 -pi / 2 < theta < 0。 
		*	转换后，直线从左过渡到右或者从右过渡到左时，角度就连续了。
		*/
		for (int i = 0; i < _lines.size(); ++i) {
			float rho = _lines[i][0], theta = _lines[i][1];
			file << rho << " " << theta << " ";
		}
		file << endl;
	}
	file.close();
}
int main()
{
//	cout << (aaa(4)) << endl;
	// Open the video file
	//cv::VideoCapture capture("../../素材/抛洒视频/MOV04216.MPG");
	//ModelingPrepare();
	//parameter
	char fileName[] = "D:\\workstation\\Traffic\\videos\\road.avi";
	//char fileName[] = "E:\\video\\GOPR9933.MP4";
	FrameProcessor *frame_processor = new LaneDetecProcessor();
	//VideoCapture capture(fileName);
	//int imageNumber(0), frameNumber(0);
	//if (!capture.isOpened())
	//{
	//	cout << "can't open video source" << endl;
	//	return 0;
	//}
	//while (true){
	//	Mat frame;
	//	capture >> frame;
	//	if (!frame.data) {
	//		cout << "can't read a frame" << endl;
	//		return 0;
	//	}
	//	cout << frameNumber++ << endl;

	//	resize(frame,frame,Size(320,240));
	//	imshow("frame",frame);
	//	char c = waitKey(10);
	//	if ('s' == c) {
	//		cv::imwrite(format("../lane_image/road/%d.jpg",imageNumber),frame);
	//		imageNumber++;
	//	}
	//}
	//KalmanTest();
	//LaneMarkingDetection(fileName);
	//process
	VideoProcessor video;
	video.setInput(fileName);
	video.setFrameProcessor( frame_processor );
//	video.displayInput("input");
	video.displayOutput("output");
	video.setDelay(1);
	video.setStartFrame(1300);
	video.run();
//	Process2Try();
//	TrainSamples();
	//SegmentSamples(capture);
	//BackgroundModeling(capture);
	//cv::Mat log = cv::imread("log.bmp",1);
	//// check if video successfully opened
	//if (!capture.isOpened())
	//	return 1;
	//// Get the frame rate
	//double rate= capture.get(CV_CAP_PROP_FPS);
	//bool stop(false);
	//cv::Mat frame; // current video frame
	//cv::namedWindow("Extracted Frame");
	//// Delay between each frame in ms
	//// corresponds to video frame rate
	//int delay= 1000/rate;
	//// for all frames in video

	//double framerate = rate;
	//cv::VideoWriter ww;
	//while (!stop) {
	//	// read next frame if any
	//	if (!capture.read(frame))
	//		break;

	//	cv::imshow("Extracted Frame",frame);
	//	// introduce a delay
	//	// or press key to stop
	//	if (cv::waitKey(delay)>=0)
	//	stop= true;
	//}
	//// Close the video file.
	//// Not required since called by destructor
	//capture.release();
}