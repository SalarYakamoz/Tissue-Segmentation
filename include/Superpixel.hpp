#pragma once
/*
Derue François-Xavier
*/

#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;


class Pixel
{
public:
	static enum State
	{
		FGND,
		BGND,
		NEUT
	};

	static enum ColorSpace
	{
		BGR,
		HSV,
		Lab,
		HDab
	};

	Point xy;
	Vec3f color;
	State state;

	Pixel():xy(Point(-1, -1)), color(Vec3f(0, 0, 0)), state(NEUT){}
	Pixel(Point xy, Vec3f color, State state = NEUT) :xy(xy), color(color), state(NEUT){}
	friend ostream& operator<<(ostream& os, const Pixel& px){os << "xy : " << px.xy << "| color : " << px.color << endl;return os;}

};
class Superpixel : public Pixel
{
public:
	static enum FeatType
	{
		HDAB,
		ORG
	};

	vector<Pixel> v_pixels;
	Mat bgrHisto;
	Mat lab_plane_lbp;
	Mat lbpHisto;
	Mat cdHisto;
	Mat image;
	Mat grayImage;
	Mat hDabImage;
	Mat hsvImage;
	Mat labels;
	Rect bounds;
	int id;
	uchar classLabel = 0;
	FeatType ft;

	Superpixel() :Pixel(){}
	Superpixel(Point xy, Vec3f color, State neut = NEUT) :Pixel(xy, color, neut){}
	
	void computeMean();
	void computeHisto(const int nBin1d);
	void computeColorDecHisto(const int nBin1d);
	void computeLBP();
	void computeBGRHisto(const int nBin1d);
	void alight(Mat& out, Vec3b color = Vec3b(255, 0, 0)) const;
	Mat getFeatMat();

};

