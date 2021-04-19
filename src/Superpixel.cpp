#include "Superpixel.hpp"
#include "funUtils.hpp"

using namespace std;
using namespace cv;

void Superpixel::computeMean() {
	xy.x = 0; xy.y = 0;
	if (v_pixels.size() != 0) {
		int minX = INT_MAX, minY = INT_MAX, maxX = 0, maxY = 0;
		for (const Pixel px : v_pixels) {
			if (px.xy.x < minX) minX = px.xy.x;
			if (px.xy.x > maxX) maxX = px.xy.x;
			if (px.xy.y < minY) minY = px.xy.y;
			if (px.xy.y > maxY) maxY = px.xy.y;
			xy += px.xy;
			color += px.color;
		}
		xy /= static_cast<float>(v_pixels.size());
		color /= static_cast<float>(v_pixels.size());
		bounds = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
	}
}

void Superpixel::computeHisto(const int nBin1d)
{
	if (ft == FeatType::ORG) {
		computeBGRHisto(nBin1d);
	}
	else {
		computeColorDecHisto(nBin1d);
	}
}

void Superpixel::computeBGRHisto(const int nBin1d)
{
	Mat pxMat(v_pixels.size(), 1, CV_32FC3);
	Vec3f* pxMat_ptr = pxMat.ptr<Vec3f>();
	for (int i = 0; i < v_pixels.size(); i++) {
		pxMat_ptr[i] = v_pixels[i].color;
	}
	CV_Assert(pxMat.isContinuous());
	funUtils::hist3D(pxMat, bgrHisto, nBin1d, funUtils::BGR);
}

void Superpixel::computeColorDecHisto(const int nBin1d)
{
	float range[] = { 0.f, 256.f };

	Mat mask = labels(bounds) == id;
	Mat hDabSrc = hDabImage(bounds);

	int histSize[] = { 16, 16 };
	const float* ranges[] = { range, range };
	int channels[] = { 0, 1 };

	calcHist(&hDabSrc, 1, channels, Mat(), cdHisto, 2, histSize, ranges, true, false);


}
void Superpixel::computeLBP()
{
	funUtils::LBPHisto(grayImage, lbpHisto, v_pixels);
}

void Superpixel::alight(Mat& out, Vec3b color) const{
	if (out.channels() == 3){
		out.setTo(color, labels == id);
	}
	else{
		out.setTo(color[0], labels == id);
	}
}

Mat Superpixel::getFeatMat()
{
	this->ft;

	Mat feats[3];
	Mat feat_total;

	feats[0] = (Mat_<float>(1, 3) << color[0], color[1], color[2]);
	normalize(feats[0], feats[0], 1, 0);
	if (ft == Superpixel::HDAB) {
		feats[1] = Mat(1, cdHisto.size[0] * cdHisto.size[1], CV_32F, cdHisto.ptr<float>());
	}
	else if (ft == Superpixel::ORG) {
		feats[1] = Mat(1, bgrHisto.size[0] * bgrHisto.size[1] * bgrHisto.size[2], CV_32F, bgrHisto.ptr<float>());
	}
	normalize(feats[1], feats[1], 1, 0);

	feats[2] = Mat(1, lbpHisto.size[0], CV_32F, lbpHisto.ptr<float>());
	normalize(feats[2], feats[2], 1, 0);

	hconcat(feats, 3, feat_total);
	return feat_total;
}

