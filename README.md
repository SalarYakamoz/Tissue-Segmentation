# Tissue-Segmentation
Image segmentation tool based on texture-aware Superpixels


![C++](https://img.shields.io/badge/C++-Solutions-blue.svg?style=flat&logo=c%2B%2B)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository implements the evaluation framework proposed in one of our research papers:
```
Fariba Damband Khameneh, Salar Razavi, Mustafa Kamasak,
Automated segmentation of cell membranes to evaluate HER2 status in whole slide images using a modified deep learning network, Computers in Biology and Medicine,
Volume 110, 2019, Pages 164-174, ISSN 0010-4825, https://doi.org/10.1016/j.compbiomed.2019.05.020.
```
## Basic Overview
The proposed approach is based on handcraft features that contribute to a supervised classifier. Classifying the regions aims to specify regions of interest for each HPF or WSI from 40X. Instead of working on all pixels, we used superpixels to compute features on meaningful and similar regions not only to increase the performance but also decrease the input
variable for the subsequent classification step. 
<p align="center"><img src="https://github.com/SalarYakamoz/Tissue-Segmentation/blob/main/images/HE_segmentation_40X.gif" width=60%>
 
 
 ## Preprocessing step
Since HER2 is associated with tumors of an epithelial region and most of the breast tumors originate in epithelial tissue, it is crucial to develop an approach
to segment different tissue structures. The proposed technique could be used on images from higher magnification level (10X) to segment regions of interest. A superpixel-based support vector machine (SVM) feature learning classifier is proposed to classify epithelial andstromal regions from features learned by textural, color, and shaoe features.
 
 <p align="center"><img src="https://github.com/SalarYakamoz/Tissue-Segmentation/blob/main/images/IHC_segmentation_10X.gif" width=60%>


## How to run it
There are two steps to run the code. First, make sure openCV_world331.dll is in your PATH enviromental (see more [Install OpenCV on Windows](https://learnopencv.com/install-opencv-4-on-windows/#:~:text=Press%20Start%2C%20type%20Environment%20variables,directory%20where%20OpenCV%20was%20installed). Make sure if you want to run the executable file you need to add OpenCV version 3.3.1. 
You can also build the code from source using Visual Studio 2017 (Other versions are not tested).

For more information, please contact [salarra67@gmail.com](mailto:salarra67@gmail.com)

Part of https://github.com/fderue/SPX_SVM is used in our project.
