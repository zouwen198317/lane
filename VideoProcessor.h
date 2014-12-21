#ifndef _GarbDetec_VideoProcessor_H_
#define _GarbDetec_VideoProcessor_H_

#include "FrameProcessor.h"

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2\contrib\contrib.hpp>

#include <string>
#include <iostream>
using std::cout;
using std::endl;

class VideoProcessor 
{
public:
	VideoProcessor() : callIt(true), delay(0), 
		fnumber(0), stop(false), frameToStop(-1) {}
	//设置处理每帧的回调函数
	void setFrameProcessor(
		void (*frameProcessingCallback)
		(cv::Mat&, cv::Mat&)) {
			process= frameProcessingCallback;
	}

	//设置FrameProcessor
	void setFrameProcessor(FrameProcessor* frameProcessorPtr)
	{
		//如果设置了FrameProcessor，则另回调函数无效
		process= 0;
		
		frameProcessor = frameProcessorPtr;
		callProcess();
	}

	//输入要处理的video的文件名，读取video文件
	bool setInput(std::string filename) {
		fnumber= 0;
		capture.release();
		return capture.open(filename);
	}

	//设置输入图像窗口
	void displayInput(std::string wn) {
		windowNameInput= wn;
		cv::namedWindow(windowNameInput);
	}
	//设置输出图像窗口
	void displayOutput(std::string wn) {
		windowNameOutput= wn;
		cv::namedWindow(windowNameOutput);
	}

	//不显示输入输出图像
	void dontDisplay() {
		cv::destroyWindow(windowNameInput);
		cv::destroyWindow(windowNameOutput);
		windowNameInput.clear();
		windowNameOutput.clear();
	}

	void initialProcessor() {
		if (!callIt)
			frameProcessor->initial();
	};

	//处理video的图像序列
	void run() {
		//当前帧
		cv::Mat frame;
		//输出图像
		cv::Mat output;
		//如果video没打开则直接返回
		if (!isOpened())
			return;
		stop= false;
		while (!isStopped()) {
			timer.reset();
			timer.start();
			// 读取下一帧
			if (!readNextFrame(frame))
				break;
			if (fnumber++ < frameToStart)
				continue;
			cout << "frame number: "<< fnumber << endl;

			// 显示输入帧
			if (windowNameInput.length()!=0) 
				cv::imshow(windowNameInput,frame);
			// 处理图像
			if (callIt) {
				if (process) //采用回调函数
					process(frame, output);
				else if (frameProcessor) //采用FrameProcessor接口
					frameProcessor->process(frame,output);
				// 显示输出图像
				if (windowNameOutput.length()!=0) 
					cv::imshow(windowNameOutput,output);
				// increment frame number

			} else {
				output= frame;
			}
			timer.stop();
			cout << "time to process frame: " << timer.getTimeMilli() << endl;
			// introduce a delay
			if (delay>=0 && cv::waitKey(delay)>=0)
				stopIt();
			// check if we should stop
			if (frameToStop>=0 && 
				getFrameNumber()==frameToStop)
				stopIt();
		}
	}
	//停止处理
	void stopIt() {
		stop= true;
	}
	// Is the process stopped?
	bool isStopped() {
		return stop;
	}
	// Is a capture device opened?
	bool isOpened() {
		return capture.isOpened();
	}
	// set a delay between each frame
	// 0 means wait at each frame
	// negative means no delay
	void setDelay(int d) {
		delay= d;
	}
	// process callback to be called
	void callProcess() {
		callIt= true;
	}
	// do not call process callback
	void dontCallProcess() {
		callIt= false;
	}

	void stopAtFrameNo(long frame) {
		frameToStop= frame;
	}
	
	void setStartFrame(long frame) {
		if (capture.isOpened())
			capture.set(CV_CAP_PROP_POS_FRAMES,frame);

		frameToStart = 0;
	}

	// return the frame number of the next frame
	long getFrameNumber() {
		// get info of from the capture device
		long fnumber= static_cast<long>(
			capture.get(CV_CAP_PROP_POS_FRAMES));
		return fnumber; 
	}

	long getFrameRate() {
		long rate = static_cast<long>(capture.get(CV_CAP_PROP_FPS));

		return rate;
	}
private:
	// to get the next frame 
	// could be: video file or camera
	bool readNextFrame(cv::Mat& frame) {
		return capture.read(frame);
	}

private:
	// OpenCV视频读取对象
	cv::VideoCapture capture;
	//用于处理每一帧的回调函数指针
	void (*process)(cv::Mat&, cv::Mat&);
	//帧处理对象指针
	FrameProcessor* frameProcessor;
	//标记是否使用了process回调函数
	bool callIt;
	//显示输入视频的窗口名字
	std::string windowNameInput;
	//显示输出视频的窗口名字
	std::string windowNameOutput;
	// 每帧间的延迟时间
	int delay;
	//已处理完的帧数
	long fnumber;
	//在该帧开始
	long frameToStart;
	//在该帧出结束
	long frameToStop;
	//结束处理的标志
	bool stop;
	//计时器
	cv::TickMeter timer;

};
#endif