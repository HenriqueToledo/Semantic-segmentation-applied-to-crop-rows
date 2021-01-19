#include <opencv4/opencv2/core/core.hpp>
#include "crop_row_detection.hpp"
#include "read_conf.hpp"
#include "auxiliary_functions.hpp"
#include "exg_otsu.hpp"
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

const string window_capture_name = "Video Capture";
const string window_capture_name1 = "ExG";
const string window_capture_name2 = "Otsu Threshold";
const string window_capture_name3 = "Morphological Transform";
const string window_capture_name4 = "Contour Detection";
const string window_capture_name5 = "Skeletonization";
const string window_capture_name6 = "Hough Lines";
const string window_capture_name7 = "Filtered Lines";

int main(int argc, char* argv[])
{
	Mat frame;
	Mat exg_output;
	Mat otsu_output;
	Mat squeezeunet;

	CropRowDetection *crd = new CropRowDetection();

	cout << endl <<  "Crop row detection program by Henrique Toledo" << endl << endl;
	
	VideoCapture cap(readConf("input"));

	while (true) {
		cap >> frame;
		if (frame.empty())
		{
			break;
		}

		cout << "Processing time frame #" << cap.get(CAP_PROP_POS_FRAMES) << ":" << endl << endl;
	
		exg_output = exG(frame);
		otsu_output = otsuThreshold(exg_output);
		crd->createImages(frame, exg_output, otsu_output);
		crd->morphologicalTransform();
		crd->skeletonization();
		crd->houghtTransform();
		crd->lineFilter();

		// Show frames
		imshow(window_capture_name, crd->images[0]->image);
		imshow(window_capture_name1, crd->images[1]->image);
		imshow(window_capture_name2, crd->images[2]->image);
		imshow(window_capture_name3, crd->images[3]->image);
		imshow(window_capture_name4, crd->images[4]->image);
		imshow(window_capture_name5, crd->images[5]->image);
		imshow(window_capture_name6, crd->images[6]->image);
		imshow(window_capture_name7, crd->images[7]->image);

		waitKey(30000);

		squeezeunet = read_squeezeunet(readConf("sample_squeeze_unet"));
		crd->createImages(frame, exg_output, squeezeunet);
		crd->morphologicalTransform();
		crd->skeletonization();
		crd->houghtTransform();
		crd->lineFilter();

		crd->compareImages(readConf("sample_ExG"), readConf("sample_squeeze_unet"), readConf("ground_truth"));

		// Show frames
		imshow(window_capture_name, crd->images[0]->image);
		imshow(window_capture_name1, crd->images[1]->image);
		imshow(window_capture_name2, crd->images[2]->image);
		imshow(window_capture_name3, crd->images[3]->image);
		imshow(window_capture_name4, crd->images[4]->image);
		imshow(window_capture_name5, crd->images[5]->image);
		imshow(window_capture_name6, crd->images[6]->image);
		imshow(window_capture_name7, crd->images[7]->image);

		char key = (char)waitKey(30000);

		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	return 0;
}