#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <omp.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat exG(Mat input){
    
    Mat exg_input = input.clone();
    Mat exg_output = Mat::zeros(exg_input.size[0], exg_input.size[1], CV_8U);
    
    int64 start = getTickCount();
    
    for(int y = 0; y < exg_input.rows; y++){
		for(int x = 0; x < exg_input.cols; x++){
            //get BGR values from pixel each interation
            Vec3b index = exg_input.at<Vec3b>(Point(x,y));
            int B = (int)index.val[0];
            int G = (int)index.val[1];
            int R = (int)index.val[2];

            // Applies the ExG Method
            exg_output.at<uchar>(Point(x,y)) = (2*G)-R-B;
            
        }
    }

    
    int64 stop = getTickCount();

    float time = (stop - start)/ getTickFrequency();
    float total_time = total_time + time;
    cout << "ExG: "<< time*1000 << " ms" << endl;

    return exg_output;

}

Mat otsuThreshold(Mat exg_outputs){

    Mat exg_output = exg_outputs.clone();
    Mat otsu_output;

    GaussianBlur(exg_output, exg_output, Size(3,3) , 0, 0);

    int64 start = getTickCount();

    int thr = threshold(exg_output, otsu_output, 254, 255, THRESH_OTSU);

    int64 stop = getTickCount();
    float time = (stop - start)/ getTickFrequency();
    float total_time = total_time + time;

    cout << "Otsu: "<< time*1000 << " ms with "<< thr << " as threshold value" <<endl;

    return otsu_output;
}