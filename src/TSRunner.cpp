#include <opencv2/highgui.hpp>
#include "SpxSvmTestEngine.hpp"

using namespace std;
using namespace cv;


void runSingle(Mat& imageTrain, Mat& imageTest, const string& modelFilePath="", const string& trainingRoisFilePath="")
{

	SpxSvmTestEngine::Settings settings;
	settings.sizeSpxOrNbSpx0 = 30;
	settings.initTypeSpx = Slic::SLIC_SIZE;
	settings.compactSpx = 35; // lower values results in irregular shapes
	settings.histNbin1d = 16;
	settings.scaleBROI = 2;
	settings.fullFrame = false;
	settings.kernelSVM = SVM::RBF;
	settings.typeSVM = SVM::C_SVC;
	//imshow("imageTrain", imageTrain);
	SpxSvmTestEngine testEngine(settings);
	if(trainingRoisFilePath.empty()) {
		if (modelFilePath.empty()) {
			testEngine.initialize(imageTrain); // train
		}
		else {
			testEngine.loadPretrainedModel(modelFilePath);
		}
	}else {
		testEngine.loadTrainInputsFromFile(imageTrain, trainingRoisFilePath);
	}
	
	testEngine.run(imageTest); // test
	testEngine.showResults();
}

int main(int argc, char** argv)
{
	//const string inputDir = "C:/pdl1_images/for-ts/";
	//const string outputDir = "C:/pdl1_images/for-ts/output/";
	//runGrid(inputDir, outputDir, 3, 4);
	const std::string filepath = argc == 2 ? argv[1] : "C:/Users/Fariba/Desktop/TS/Capture2.PNG";
	Mat train = imread(filepath, IMREAD_COLOR);
	//imshow("train", train);
	Mat test = train.clone();
	//runSingle(train, test, "", "training_selections.txt");
	runSingle(train, test, "", "");
	
	waitKey(0);
	cin.get();
	return 0;
}
