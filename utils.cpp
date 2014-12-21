/*************************************************************************
    > File Name: utils.cpp
    > Author: onerhao
    > Mail: haodu@hustunique.com
    > Created Time: Sun 17 Mar 2013 09:53:47 PM CST
 ************************************************************************/

#include "utils.h"

Line::Line()
    :p0(Point2d(0,0)),p1(Point2d(0,0)),angle(0),k(0),b(0),length(0) {}

Line::Line(Point2d p0,Point2d p1)
  :p0(p0),p1(p1),votes(0),visited(0),found(0)
{
  angle=calSlopeAngle(p0,p1);
  k=calSlope(p0,p1);
  b=p0.y-k*p0.x;
  length=calLength(p0,p1);
  dis2origin = fabs(b) / sqrt(k * k + 1);
}

Line::Line(float k,float b)
  :p0(Point2d(0,0)),p1(Point2d(0,0)),angle(atan(k)),k(k),b(b),
  length(0),votes(0),visited(false), found(false){}

void Line::draw(cv::Mat& img, cv::Scalar color)
{
	assert(img.type() == CV_8UC3);

	if ( Point2d(0,0) == p0 && Point2d(0,0) == p1){
		p0.y = 0;
		p1.y = img.rows;
		p0.x = this->getx(0);
		p1.x = this->getx(img.rows);
	}
	line( img, p0, p1, color, 2, 8 );
}

double Line::getx(float y)
{
    float x=(y-this->b)/this->k;
    return x;
}

double Line::gety(float x)
{
    float y=this->k*x+this->b;
    return y;
}

Line::~Line(){}

double calSlope(Point2d p0,Point2d p1)
{
    int dx=p0.x-p1.x;
    int dy=p0.y-p1.y;
    double slope;

    if(dy==0)
        return 0;
    else if(dx==0)
        return 0;
    else
        slope=dy/(double)dx;

    return slope;
}

double calSlopeAngle(Point2d p0,Point2d p1)
{
    //return the angle of slope in radius
    double theta=0;

    int dx=p0.x-p1.x;
    int dy=p0.y-p1.y;

    if(dx==0)
    {
        theta=CV_PI/2;
    }
    else if(dy==0)
    {
        theta=0;
    }
    else
    {
        theta=atan(dy/(double)dx);
    }
    if(theta<0)
        theta=CV_PI + theta;
    return theta;
}

double calIntercept(Point2d p0,Point2d p1)
{
    int k=calSlope(p0,p1);
    int b=p0.y-k*p0.x;
    return b;
}

double calLength(Point2d p0,Point2d p1)
{
    int dx=p0.x-p1.x;
    int dy=p0.y-p1.y;

    return sqrt((float)dx*dx+dy*dy);
}

Point2d midPoint(Point2d p0,Point2d p1)
{
    return Point2d((p0.x+p1.x)/2,(p0.y+p1.y)/2);
}

Point2d calIntersection(Line l0,Line l1)
{
    Point2d intersection;
    intersection.x=(l1.b-l0.b)/(l0.k-l1.k);
    intersection.y=l0.k*intersection.x+l0.b;

    return intersection;
}

int pointIn(Point2d pt,std::vector<Point2d>& pts)
{
    //sort(pts,
    return 0;
}

