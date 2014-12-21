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
	//���ô���ÿ֡�Ļص�����
	void setFrameProcessor(
		void (*frameProcessingCallback)
		(cv::Mat&, cv::Mat&)) {
			process= frameProcessingCallback;
	}

	//����FrameProcessor
	void setFrameProcessor(FrameProcessor* frameProcessorPtr)
	{
		//���������FrameProcessor������ص�������Ч
		process= 0;
		
		frameProcessor = frameProcessorPtr;
		callProcess();
	}

	//����Ҫ�����video���ļ�������ȡvideo�ļ�
	bool setInput(std::string filename) {
		fnumber= 0;
		capture.release();
		return capture.open(filename);
	}

	//��������ͼ�񴰿�
	void displayInput(std::string wn) {
		windowNameInput= wn;
		cv::namedWindow(windowNameInput);
	}
	//�������ͼ�񴰿�
	void displayOutput(std::string wn) {
		windowNameOutput= wn;
		cv::namedWindow(windowNameOutput);
	}

	//����ʾ�������ͼ��
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

	//����video��ͼ������
	void run() {
		//��ǰ֡
		cv::Mat frame;
		//���ͼ��
		cv::Mat output;
		//���videoû����ֱ�ӷ���
		if (!isOpened())
			return;
		stop= false;
		while (!isStopped()) {
			timer.reset();
			timer.start();
			// ��ȡ��һ֡
			if (!readNextFrame(frame))
				break;
			if (fnumber++ < frameToStart)
				continue;
			cout << "frame number: "<< fnumber << endl;

			// ��ʾ����֡
			if (windowNameInput.length()!=0) 
				cv::imshow(windowNameInput,frame);
			// ����ͼ��
			if (callIt) {
				if (process) //���ûص�����
					process(frame, output);
				else if (frameProcessor) //����FrameProcessor�ӿ�
					frameProcessor->process(frame,output);
				// ��ʾ���ͼ��
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
	//ֹͣ����
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
	// OpenCV��Ƶ��ȡ����
	cv::VideoCapture capture;
	//���ڴ���ÿһ֡�Ļص�����ָ��
	void (*process)(cv::Mat&, cv::Mat&);
	//֡�������ָ��
	FrameProcessor* frameProcessor;
	//����Ƿ�ʹ����process�ص�����
	bool callIt;
	//��ʾ������Ƶ�Ĵ�������
	std::string windowNameInput;
	//��ʾ�����Ƶ�Ĵ�������
	std::string windowNameOutput;
	// ÿ֡����ӳ�ʱ��
	int delay;
	//�Ѵ������֡��
	long fnumber;
	//�ڸ�֡��ʼ
	long frameToStart;
	//�ڸ�֡������
	long frameToStop;
	//��������ı�־
	bool stop;
	//��ʱ��
	cv::TickMeter timer;

};
#endif