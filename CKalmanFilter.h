#include <iostream>
#include <vector>

#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

class CKalmanFilter
{
public:
	CKalmanFilter();
	CKalmanFilter(vector<Vec2f>);
	CKalmanFilter(vector<Line>&);
	~CKalmanFilter();

	void initial(const Vec2f&, const Vec2f&);

	vector<Vec2f> update(vector<Vec2f>);
	void update(const Vec2f& left, const Vec2f& right);
	vector<Line> update(vector<Line>&);
	void changeState(const Vec2f&, const Vec2f& );
	void getState(Vec2f&, Vec2f&);
	void getPreState(Vec2f&, Vec2f&);
	bool isInitialing();

	//kalman filter failed to update for REJECT_THRESH number frames.
	bool isStoped();

private:
	enum {
		REJECT_THRESH = 10,
		ACCEPT_THRESH = 5,
	};

	KalmanFilter* kalman;
	int rejectNumber;
	int acceptNumber;
	bool _isInitialing;

};