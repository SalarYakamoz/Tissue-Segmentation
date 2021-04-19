#include <opencv2/opencv.hpp>
#include "SpxSvmTestEngine.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;


void combineTiles(const string &inputDir, const string &outputDir, int rows, int cols, int scaleFactor)
{
	/*if(rows % scaleFactor != 0 || cols % scaleFactor != 0) {
		cout << "Scale factor not divisible by dimensions!" << endl;
		return;
	}*/
	Size tileSize;
	for (int i = 0; i < rows / scaleFactor; i++) {
		for (int j = 0; j < cols / scaleFactor; j++) {
			Mat newTile;
			vector<Mat> tileRows;
			for (int l = 0; l < scaleFactor; l++) {
				Mat tileRow;
				vector<Mat> hTiles;
				for (int k = 0; k < scaleFactor; k++) {
					Mat tile = imread(inputDir + to_string(i * scaleFactor * cols + j * scaleFactor + l * cols + k) + ".png");
					if(tile.empty()) {
						tile = Mat(tileSize, CV_8UC3, Scalar(255, 255, 255));
					} else {
						tileSize = tile.size();
					}

					resize(tile, tile, Size(0, 0), 1. / scaleFactor, 1. / scaleFactor, INTER_AREA);
					hTiles.push_back(tile);
				}
				hconcat(hTiles, tileRow);
				tileRows.push_back(tileRow);
			}
			vconcat(tileRows, newTile);
			imwrite(outputDir + to_string(i*(cols / scaleFactor) + j) + ".png", newTile);
		}
		cout << "Row: " << i << endl;
	}
}


void combineTiles(string inputDir, Mat &combined, int rows, int cols)
{
	vector<Mat> tileRows;
	for(int i = 0; i < rows; i++) {
		vector<Mat> hTiles;
		Mat tileRow;
		for (int j = 0; j < cols; j++) {
			Mat tile = imread(inputDir + to_string(i*cols + j) + ".png");
			Mat smallTile;
			Size smallSize(tile.cols/16, tile.rows/16);
			resize(tile, smallTile, smallSize, 0, 0, INTER_AREA);
			hTiles.push_back(smallTile);
		}
		hconcat(hTiles, tileRow);
		tileRows.push_back(tileRow);
		cout << "row: " << i << endl;
	}
	vconcat(tileRows, combined);
	cout << "Combined size: " << combined.size() << endl;
	imwrite(inputDir + "combined.png", combined);
}

void runSingle(Mat& imTrain, Mat& imTest)
{
	//test
	SpxSvmTestEngine::Settings settings;
	settings.sizeSpxOrNbSpx0 = 30; // what are they??? (setting options)
	settings.initTypeSpx = Slic::SLIC_SIZE;
	settings.compactSpx = 70; // lower values results in irregular shapes
	settings.histNbin1d = 16;
	settings.scaleBROI = 2;
	settings.fullFrame = false;
	settings.kernelSVM = SVM::RBF;
	settings.typeSVM = SVM::C_SVC;

	SpxSvmTestEngine testEngine(settings);
	testEngine.initialize(imTrain); // train
	testEngine.run(imTest); // test
	testEngine.showResults();
}


int main(int argc, char** argv)
{
	Mat test = imread("C:/her2_images/104/combined16.png");
	resize(test, test, Size(0, 0), 0.5, 0.5, INTER_AREA);
	Mat train = test(Rect(test.cols/4, test.rows/4, test.cols/2, test.rows/2)).clone();
	runSingle(train, test);
	//combineTiles("C:/her2_images/104/", "C:/her2_images/104-far8/", 82, 66, 8);
}