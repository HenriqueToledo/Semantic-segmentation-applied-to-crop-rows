#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/*
 Some useful functions
*/

Mat read_squeezeunet(string path_to_input){

    Mat squeezeunet = imread(path_to_input, 0);
    threshold(squeezeunet, squeezeunet, 200, 255, THRESH_BINARY);

    return squeezeunet;
}

void augmentation(string path_to_input, string path_to_output){

    Mat input = imread(path_to_input);
    Mat HLS_image;
    Mat output;
  
    cvtColor(input, HLS_image, COLOR_RGB2HLS);

    imshow("Teste", HLS_image);

    waitKey(10000);

    float brightness = 0.5;

    for(int y = 0; y < HLS_image.rows; y++){
		for(int x = 0; x < HLS_image.cols; x++){

        Vec3b index = HLS_image.at<Vec3b>(Point(x,y));
        int H = (int)index.val[0];
        int L = (int)index.val[1];
        int S = (int)index.val[2];

        HLS_image.at<uchar>(Point(x,y)) = H+L+S;

        if(HLS_image.at<uchar>(Point(x,y)) > 255){
            HLS_image.at<uchar>(Point(x,y)) = 255;
        }

        }
    }

    imshow("Teste", HLS_image);

    waitKey(10000);  

    cvtColor(HLS_image, output , COLOR_HLS2RGB);

    imwrite(path_to_output, output);
}

void binarize_image(string path_to_input, string path_to_output){

    Mat input = imread(path_to_input);
    Mat output = imread(path_to_output);

    threshold(input, output, 120, 255, THRESH_BINARY);

    imwrite(path_to_output, output);
}

void morphologicalTransform(string path_to_input, string path_to_output, string mode){

    Mat input = imread(path_to_input);
    Mat output = imread(path_to_output);

    if(mode == "opening"){
        erode(input, output, Mat(), Point(-1,-1), 1, 1, 1);
        dilate(output, output, Mat(), Point(-1,-1), 3, 1, 1);
    }
    else if(mode == "closing"){
        dilate(input, output, Mat(), Point(-1,-1), 5, 1, 1);
        erode(output, output, Mat(), Point(-1,-1), 6, 1, 1);
    }
    else if(mode == "dilate"){
        dilate(input, output, Mat(), Point(-1,-1), 3, 1, 1);
    }
    else{
        erode(input, output, Mat(), Point(-1,-1), 1, 1, 1);
    }

    imwrite(path_to_output, output);
}

void print_as_matrix(string path_to_sample){

    Mat sample = imread(path_to_sample, 0);

    if(sample.empty()){

        cout << "Could not find the images, please check the path!" << endl;
        return;
    }


    cout << sample << endl;
    
    cout << endl;
}


void histogram(string path_to_input, string path_to_output){

    Mat input = imread(path_to_input);
    Mat histogram;

    vector<Mat> grayscale_plane;
    split( input, grayscale_plane );
    
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    calcHist(&grayscale_plane[0], 1, 0, Mat(), histogram, 1, &histSize, &histRange, true, false);
    
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(histogram.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
    }

    imshow("Histogram", histImage );
    waitKey();
}