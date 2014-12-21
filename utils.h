/*************************************************************************
    > File Name: utils.h
    > Author: onerhao
    > Mail: haodu@hustunique.com
    > Created Time: Sun 17 Mar 2013 10:56:35 PM CST
 ************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <opencv2\core\core.hpp>
using cv::Point2d;

class Line
{
	public:
		Point2d p0,p1;
		float angle,k,b,length,dis2origin;
		int votes;
		bool visited,found;

	public:
		Line();
		Line(Point2d p0,Point2d p1);
		Line(float k,float b);
		virtual double getx(float y);
		virtual double gety(float x);

		void draw(cv::Mat& img, cv::Scalar color = cv::Scalar(255,0,0)); 
		bool operator<(const Line &l) const
		{
			return (this->length < l.length);
		}
		virtual ~Line();
};

//return slope of the line formed by two points
extern double calSlope(Point2d p0,Point2d p1);

//return the slope angle of the line formed by two points
extern double calSlopeAngle(Point2d p0,Point2d p1);

//return the intercept of line y=k*x+b
extern double calIntercept(Point2d p0,Point2d p1);

//return of the lenght of two points
extern double calLength(Point2d p0,Point2d p1);

//return the middle point
extern Point2d midPoint(Point2d p0,Point2d p1);

//return intersection of two lines
extern Point2d calIntersection(Line l0,Line l1);

//check whether a point is in the *** formed by the pts
extern int pointIn(Point2d pt,std::vector<Point2d>& pts);

#endif
