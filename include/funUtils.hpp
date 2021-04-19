#pragma once
#include <opencv2/opencv.hpp>

class Pixel;

using namespace std;
using namespace cv;

namespace funUtils {

	enum HistColor {
		BGR,
		HSV,
		Lab
	};

	void getGrabCutSeg(const Mat& inIm, Mat& mask_fgnd, Rect ROI);
	cv::Mat makeMask(Rect ROIin, int wFrame, int hFrame, float scale = 2, bool fullFrame = false);
	void adaptROI(Rect& ROI, int wFrame, int hFrame);
	void hist3D(Mat& image, Mat& hist, int Nbin, HistColor histColorSpace);
	void LBPbasic(Mat &src, Mat& dst, vector<Pixel> &locations);
	void LBP(Mat &src, Mat &dst);
	void LBPHisto(Mat &src, Mat& hist, vector<Pixel> &locations);
}
