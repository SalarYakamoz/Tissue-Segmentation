/*
Objectif : Test SVM binary classification of foreground/background based on Superpixel
input : 
- Training image : image used to train the svm classifier
- ROI : bounding box splitting the training image. Inside of ROI is foreground (positive sample), outside is background (negative sample)
- TestImage : apply the classifier on the superpixel found in TestImage
output : Classification of the Superpixels extracted from TestImage

ex : 
SpxSvmTestEngine::Settings settings;
SpxSvmTestEngine testEngine;
testEngine.initialize(imTrain, ROImouse, settings);
testEngine.run(imTest);
testEngine.showResults(imTrain,imTest);

author : Derue François-Xavier
francois.xavier.derue<at>gmail.com
*/
#pragma once
#ifdef MAKEDLL
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "Slic.hpp"
#include "Superpixel.hpp"
#include <memory>


using namespace std;
using namespace cv;
using namespace cv::ml;

void makeSpxVec(Slic& slic, vector<Superpixel>& v_spx, Mat& imageBGR8, const int nBin1d = 6);
void trainSVM(Ptr<SVM>& svm, vector<Superpixel>& v_spx, SVM::Types svmType, SVM::KernelTypes kernelType);
Mat createFeatMat(vector<Superpixel*>& v_spx);
Mat computeLabelsMat(const vector<Superpixel*>& v_spx);

class SpxSvmTestEngine{
public:
	struct Settings
	{
		int sizeSpxOrNbSpx0 = 6;
		Slic::InitType initTypeSpx = Slic::SLIC_SIZE;
		int compactSpx = 35;
		int histNbin1d = 6;
		int scaleBROI = 2;
		bool fullFrame = false;

		SVM::KernelTypes kernelSVM = SVM::RBF;
		SVM::Types typeSVM = SVM::C_SVC;
	};
	static const string trainingWindowName;
	static const string testWindowName;

	Mat getTrainLabels() const { return m_pSlicTrain->getLabels(); }
	Size getImageSize() const { return m_ImTrain.size(); }
	bool setSpxLabel(int i, int label, Vec3b overlayColor);
	void showImTrain() const;
	void showImTest() const;
	void toggleTestOverlay() { m_ShowTestOverlay = !m_ShowTestOverlay; }
	EXPORT SpxSvmTestEngine(const Settings& settings);
	~SpxSvmTestEngine(){}

	EXPORT void loadTrainInputsFromFile(Mat& imTrain, const std::string &inputPath);
	EXPORT void loadPretrainedModel(const std::string &inputPath);
	EXPORT void initialize(Mat& imTrain);
	EXPORT void run(Mat& imTest);
	EXPORT void showResults();
	
private:
	unique_ptr<Slic> m_pSlicTrain;
	unique_ptr<Slic> m_pSlicTest;
	Ptr<SVM> m_SvmClassifier;
	vector<Superpixel> vSpxTrain;
	vector<Superpixel> vSpxTest;
	Settings m_Settings;
	Mat m_ImTrain;
	Mat m_ImTrainOverlay;
	Mat m_ImTest;
	Mat m_ImTestOverlay;
	bool m_ShowTestOverlay = true;
};