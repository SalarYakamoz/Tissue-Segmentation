#include "SpxSvmTestEngine.hpp"
#include <iostream>
#include <fstream>
#include "Slic.hpp"
#include "funUtils.hpp"
#include "LambdaParallel.hpp"

const string SpxSvmTestEngine::trainingWindowName = "Training selection";
const string SpxSvmTestEngine::testWindowName = "Test results";


void onImTestMouse(int event, int x, int y, int flags, void* data)
{
	if (event != EVENT_LBUTTONDOWN && event != EVENT_LBUTTONDBLCLK)
		return;

	SpxSvmTestEngine* engine = static_cast<SpxSvmTestEngine*>(data);
	engine->toggleTestOverlay();
	engine->showImTest();
}

void onImTrainMouse(int event, int x, int y, int flags, void* data)
{
	if (event != EVENT_LBUTTONDOWN && event != EVENT_RBUTTONDOWN && event != EVENT_MOUSEMOVE && event != EVENT_MBUTTONDOWN)
		return;

	SpxSvmTestEngine* engine = static_cast<SpxSvmTestEngine*>(data);
	const Size imSize = engine->getImageSize();
	if (x < 0 || y < 0 || x >= imSize.width || y >= imSize.height)
		return;

	Mat trainLabels = engine->getTrainLabels();
	const int label = trainLabels.at<int>(y, x);
	bool needRedraw = false;
	if (event == EVENT_MBUTTONDOWN) {
		needRedraw = engine->setSpxLabel(label, 3, Vec3b(255, 0, 0));
	}
	else if (event == EVENT_LBUTTONDOWN) {
		needRedraw = engine->setSpxLabel(label, 1, Vec3b(0, 255, 0));
	}
	else if (event == EVENT_RBUTTONDOWN) {
		needRedraw = engine->setSpxLabel(label, 2, Vec3b(0, 0, 255));
	}
	else if (event == EVENT_MOUSEMOVE) {
		if ((flags & EVENT_FLAG_LBUTTON) > 0) {
			needRedraw = engine->setSpxLabel(label, 1, Vec3b(0, 255, 0));
		}
		if ((flags & EVENT_FLAG_RBUTTON) > 0) {
			needRedraw = engine->setSpxLabel(label, 2, Vec3b(0, 0, 255));
		}
		if ((flags & EVENT_FLAG_MBUTTON) > 0) {
			needRedraw = engine->setSpxLabel(label, 3, Vec3b(255, 0, 0));
		}
	}

	if (needRedraw) engine->showImTrain();
}

void makeSpxVec(Slic& slic, vector<Superpixel>& v_spx, Mat& imageBGR8, const int nBin1d)
{
	const int Nspx = slic.getNbSpx();
	v_spx.resize(Nspx);
	const Mat labels = slic.getLabels();
	for (int i = 0; i < labels.rows; i++) {
		const Vec3b* image_ptr = imageBGR8.ptr<Vec3b>(i);
		const int* label_ptr = labels.ptr<int>(i);
		for (int j = 0; j < labels.cols; j++) {
			v_spx[label_ptr[j]].v_pixels.push_back(Pixel(Point(j, i), Vec3f(image_ptr[j])));
		}
	}
	Mat grayImage;

	cvtColor(imageBGR8, grayImage, CV_BGR2GRAY);

	parallel_for(Range(0, Nspx), [&](const Range& range) {
		for (int i = range.start; i < range.end; i++) {
			v_spx[i].ft = Superpixel::ORG;
			v_spx[i].image = imageBGR8;
			v_spx[i].grayImage = grayImage;
			v_spx[i].id = i;
			v_spx[i].labels = slic.getLabels();
			v_spx[i].computeMean();
			v_spx[i].computeHisto(nBin1d);
			v_spx[i].computeLBP();
		}
	});

}

bool SpxSvmTestEngine::setSpxLabel(int i, int label, Vec3b overlayColor)
{
	if(vSpxTrain[i].classLabel == label) {
		return false;
	}
	vSpxTrain[i].classLabel = label;

	vSpxTrain[i].alight(m_ImTrainOverlay, overlayColor);
	return true;
}

void SpxSvmTestEngine::showImTrain() const
{
	Mat result;
	Mat overlayMask;
	cvtColor(m_ImTrainOverlay, overlayMask, CV_BGR2GRAY);
	overlayMask = ~(overlayMask>0);

	addWeighted(m_ImTrain, 0.5, m_ImTrainOverlay, 0.5, 0, result);
	m_ImTrain.copyTo(result, overlayMask);
	imshow(trainingWindowName, result);
}

Mat computeLabelsMat(const vector<Superpixel*>& v_spx)
{
	Mat labelsMat(v_spx.size(), 1, CV_32SC1, Scalar(0));
	int* labelsMat_ptr = (int*)labelsMat.data;
 	for (int i = 0; i < v_spx.size(); i++) {
		 if(v_spx[i]->classLabel != 0) {
			 labelsMat_ptr[i] = v_spx[i]->classLabel;
		 }
		/*if (v_spx[i]->state == Pixel::FGND) labelsMat_ptr[i] = 1;
		else if (v_spx[i]->state == Pixel::BGND) labelsMat_ptr[i] = 2;*/
		else cerr << "error : no NEUT is allowed when computing labels Mat " << endl;
	}
	return labelsMat;
}

Mat createFeatMat(vector<Superpixel*>& v_spx)
{
	CV_Assert(!v_spx.empty());

	Mat featsMat;
	vector<Mat> featMats;
	featMats.reserve(v_spx.size());
	for (int i = 0; i < v_spx.size(); i++) {
		featMats.push_back(v_spx[i]->getFeatMat());
	}
	vconcat(featMats, featsMat);

	return featsMat;
}

void trainSVM(Ptr<SVM>& svm, vector<Superpixel>& v_spx, SVM::Types svmType, SVM::KernelTypes kernelType)
{
	CV_Assert(!v_spx.empty());
	svm->setType(svmType);
	svm->setKernel(kernelType);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	vector<Superpixel*> v_spx_ptr;
	for (int i = 0; i < v_spx.size(); i++) {
		if (v_spx[i].classLabel != 0) {
			v_spx_ptr.push_back(&v_spx[i]);
		}
	}
	const Mat fbSpxFeatMat = createFeatMat(v_spx_ptr);
	const Mat labelsMat = computeLabelsMat(v_spx_ptr);
	const Ptr<TrainData> tdata = TrainData::create(fbSpxFeatMat, ROW_SAMPLE, labelsMat);
	svm->trainAuto(tdata); // optimize parameter with k-fold

}
static float predictSVM(Ptr<SVM>& svm, Superpixel& spx)
{
	Mat spxTest = spx.getFeatMat();
	Mat out;
	//cout << "spxTest: " << spxTest << endl;
	return svm->predict(spxTest);
	//out,StatModel::RAW_OUTPUT);
    //return out.at<float>(0,0);
}

SpxSvmTestEngine::SpxSvmTestEngine(const Settings& settings)
{
	m_pSlicTrain = make_unique<Slic>();
	m_pSlicTest = make_unique<Slic>();
	m_SvmClassifier = SVM::create();
	m_Settings = settings;
}

void SpxSvmTestEngine::initialize(Mat& imTrain)
{
	CV_Assert(imTrain.data != nullptr);

	m_ImTrain = imTrain;
	//imshow("m_ImTrain", m_ImTrain);
	m_ImTrainOverlay = Mat::zeros(imTrain.size(), CV_8UC3);
	// m_FgndROI = ROI;

	//Superpixel segmentation
	m_pSlicTrain->initialize(imTrain, m_Settings.sizeSpxOrNbSpx0, m_Settings.compactSpx, 10, m_Settings.initTypeSpx);
	m_pSlicTrain->generateSpx(imTrain);

	//create Superpixel vector from slic
	makeSpxVec(*m_pSlicTrain, vSpxTrain, imTrain);  // creates super pixel and find their features

	//set Fgnd and Bgnd Superpixel
	m_pSlicTrain->display_contours(imTrain);

	imshow(trainingWindowName, imTrain);
	setMouseCallback(trainingWindowName, onImTrainMouse, static_cast<void*>(this));
	waitKey();
	setMouseCallback(trainingWindowName, nullptr);

	// //save inputs
	// ofstream ofs("training_selections.txt");
	// for(int i=0; i<vSpxTrain.size(); i++) {
	// 	//if(vSpxTrain[i].state == Pixel::BGND) {
	// 	if(vSpxTrain[i].classLabel == 2) {
	// 		ofs << i << endl;
	// 	}
	// }
	// ofs << "hede" << endl;
	// for(int i=0; i<vSpxTrain.size(); i++) {
	// 	if(vSpxTrain[i].classLabel == 1) {
	// 		ofs << i << endl;
	// 	}
	// }
	// ofs.close();

	//train a classifier
	trainSVM(m_SvmClassifier, vSpxTrain, m_Settings.typeSVM, m_Settings.kernelSVM);
	m_SvmClassifier->save("svm.xml");
}

void SpxSvmTestEngine::loadTrainInputsFromFile(Mat& imTrain, const std::string &inputPath)
{
	cout << "Loading training inputs from file: " << inputPath << endl;
	CV_Assert(imTrain.data != nullptr);

	m_ImTrain = imTrain;
	m_ImTrainOverlay = Mat::zeros(imTrain.size(), CV_8UC3);
	// m_FgndROI = ROI;

	//Superpixel segmentation
	m_pSlicTrain->initialize(imTrain, m_Settings.sizeSpxOrNbSpx0, m_Settings.compactSpx, 10, m_Settings.initTypeSpx);
	m_pSlicTrain->generateSpx(imTrain);

	//create Superpixel vector from slic
	makeSpxVec(*m_pSlicTrain, vSpxTrain, imTrain);  // creates super pixel and find their features

	ifstream ifs(inputPath);
	string line;
	if (ifs.is_open()) {
		bool isFgnd = false;
		while (getline(ifs, line)) {
			if(line == "hede") {
				isFgnd = true;
			}else {
				if(isFgnd) {
					vSpxTrain[atoi(line.c_str())].classLabel = 1;
				}else {
					vSpxTrain[atoi(line.c_str())].classLabel = 2;
				}
			}
		}
		ifs.close();
	}

	trainSVM(m_SvmClassifier, vSpxTrain, m_Settings.typeSVM, m_Settings.kernelSVM);
}

void SpxSvmTestEngine::loadPretrainedModel(const std::string &inputPath)
{
	cout << "Loading pre-trained model from file: " << inputPath << endl;
	m_SvmClassifier = Algorithm::load<SVM>(inputPath);
}

void SpxSvmTestEngine::run(Mat& imTest)
{
	CV_Assert(imTest.data != nullptr);

	m_ImTest = imTest;
	m_ImTestOverlay = Mat::zeros(imTest.size(), CV_8UC3);

	m_pSlicTest->initialize(imTest, m_Settings.sizeSpxOrNbSpx0, m_Settings.compactSpx, 10, m_Settings.initTypeSpx);
	m_pSlicTest->generateSpx(imTest);

	makeSpxVec(*m_pSlicTest, vSpxTest, imTest);

	//classify all spx (or just in a search area where the samples will be)
	
	for (int i = 0; i < vSpxTest.size(); i++) {
		float response = predictSVM(m_SvmClassifier, vSpxTest[i]);
		vSpxTest[i].classLabel = static_cast<uchar>(static_cast<int>(response));
	}

}

void SpxSvmTestEngine::showResults()
{
	//classification result
	for (int i = 0; i < vSpxTest.size(); i++) {
		if (vSpxTest[i].classLabel == 1)vSpxTest[i].alight(m_ImTestOverlay, Vec3b(0, 255, 0));
		else if (vSpxTest[i].classLabel == 2)vSpxTest[i].alight(m_ImTestOverlay, Vec3b(0, 0, 255));
		else if (vSpxTest[i].classLabel == 3)vSpxTest[i].alight(m_ImTestOverlay, Vec3b(255, 0, 0));
	}
	m_pSlicTest->display_contours(m_ImTest);
	showImTest();
	setMouseCallback(testWindowName, onImTestMouse, static_cast<void*>(this));

	waitKey();
}

void SpxSvmTestEngine::showImTest() const
{
	if(m_ShowTestOverlay) {
		Mat result;
		Mat overlayMask;
		cvtColor(m_ImTestOverlay, overlayMask, CV_BGR2GRAY);
		overlayMask = ~(overlayMask>0);

		addWeighted(m_ImTest, 0.5, m_ImTestOverlay, 0.5, 0, result);
		m_ImTest.copyTo(result, overlayMask);
		imshow(testWindowName, result);
	}
	else {
		imshow(testWindowName, m_ImTest);
	}
}