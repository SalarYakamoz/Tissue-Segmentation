/*
SLIC : CPU version
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

This class implement the superpixel segmentation "SLIC Superpixels",
Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
EPFL Technical Report no. 149300, June 2010.
Copyright (c) 2010 Radhakrishna Achanta [EPFL]. All rights reserved.

*/
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

typedef std::vector<std::vector<int> > vec2di;

struct center;
class Slic
{
public:

	Slic(){}
	~Slic(){}

	static enum InitType{
		SLIC_SIZE,
		SLIC_NSPX
	};

	/*SLIC_SIZE -> initialize by specifying the spx size
	SLIC_NSPX -> initialize by specifying the number of spx*/
	void initialize(const cv::Mat& frame, const int nspx, const float wc, const int nIteration, const Slic::InitType type);
	void generateSpx(const cv::Mat& frame);

	cv::Mat getLabels(){ return m_labels; }
	int getNbSpx(){ return m_nSpx; }
	int getSpxDiam(){ return m_diamSpx; }


	void display_contours(cv::Mat& image, cv::Scalar colour = cv::Scalar(255, 0, 0));
	void displayMeanColor(cv::Mat& out);

private:
	int m_nSpx;
	int m_nspx_1;
	float m_wc;
	int m_width;
	int m_height;
	int m_diamSpx;
	int m_nIteration;
	cv::Mat m_labels;
	std::vector<std::vector<float> > m_allDist;
	std::vector<center> m_allCenters;
	std::vector<center> m_allCenters_1;

	void resetVariables();
	void enforceConnectivity(cv::Mat& frame);
	void findCenters(cv::Mat& frame);
	void updateCenters(cv::Mat& frame);

};

struct center
{
	cv::Point xy;
	float Lab[3];
	center() :xy(cv::Point(0, 0)){
		Lab[0] = 0.f;
		Lab[1] = 0.f;
		Lab[2] = 0.f;
	}
};



