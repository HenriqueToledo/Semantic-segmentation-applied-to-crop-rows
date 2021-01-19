#include <opencv4/opencv2/imgproc/imgproc_c.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <vector>
#include <omp.h>
#include <iostream>

using namespace cv;
using namespace std;

class Line{
public:
	Point pt1;
    Point pt2;
    
    Line(Point pt1, Point pt2){
	    this->pt1 = pt1;
	    this->pt2 = pt2;
	}
};

class Image{
public:
    Mat image;
    string name;
};

class CropRowDetection{
private:

    // Processing time
    int64 start = 0;
    int64 stop = 0;
    float time = 0;
    float total_time = 0;
    
    int height;
    int width;
    vector <Line*>lines;

    //methods
    Point intersectPoint(Point pt1, Point pt2, Point p3, Point p4);
    void processingTime(string flag);

public:

    Image *images[7];

    //main methods
    CropRowDetection();
    void createImages(Mat image, Mat inputGrayscale, Mat inputBinary);
    void morphologicalTransform();
    void skeletonization();
    void houghtTransform();
    void lineFilter();
    void compareImages(string sample_ExG, string sample_squeeze_unet, string path_to_ground_truth);
};

CropRowDetection::CropRowDetection(){

    // load the images from the algorithm
    for(int i=0; i < 8; i++){
        this->images[i] = new Image();
    }
}

void CropRowDetection::createImages(Mat frame, Mat exg_output, Mat binary_output){

    //get image size
    this->width = frame.size[1];
    this->height = frame.size[0];

    //create the images
    this->images[0]->image = frame.clone();

    this->images[1]->image = exg_output.clone();

    this->images[2]->image = binary_output.clone();

    this->images[3]->image = Mat::zeros(this->height, this->width, CV_8U);

    this->images[4]->image = Mat::zeros(this->height, this->width, CV_8U);

    this->images[5]->image = Mat::zeros(this->height, this->width, CV_8U);

    this->images[6]->image = frame.clone();

    this->images[7]->image = frame.clone();
    
}

void CropRowDetection::morphologicalTransform(){

    this->processingTime("start");

    erode(this->images[2]->image, this->images[3]->image, Mat(), Point(-1,-1), 1, 1, 1);
    dilate(this->images[3]->image, this->images[3]->image, Mat(), Point(-1,-1), 1, 1, 1);

    this->processingTime("measure");
    cout << "Morphological Transform: "<< this->time*1000 << " ms" << endl;
}

void CropRowDetection::skeletonization(){

    this->processingTime("start");

    int y, x;
    int height = this->height-1; 
    int width = this->width-1;

    for(y=(height); y >= 0; y--){
        
        bool isLine = false;
        int x1,x2 = 0;
        
	    for(x=(width); x >= 0; x--){
            
            // as soon as it detects the first white pixel, it considers the right border and saves its value
            if(isLine == false && this->images[3]->image.at<uchar>(Point(x,y)) == 255){
                x1 = x2 = x;
                isLine = true;
            }
            // move the left limit until you find the next black pixel
            else if(isLine == true && this->images[3]->image.at<uchar>(Point(x,y)) == 255){
                x2 = x;
            }
            // as soon as it finds the black pixel, assign the value 255 (white) to the x1 and x2 boundaries of the contour
            else if(isLine == true && this->images[3]->image.at<uchar>(Point(x,y)) == 0){
                this->images[4]->image.at<uchar>(Point(x1,y)) = 255; //x1 é o limite direito
                this->images[4]->image.at<uchar>(Point(x2,y)) = 255; //x2 é o limite esquerdo
                if(x2 != 0){
                    // averages the contours and then applies skeletonization
                    int meanx = (x1+x2) / 2;
                    this->images[5]->image.at<uchar>(Point(meanx,y)) = 255;
                }
                isLine = false;
                x1 = x2 = 0;
            }
        }
    }
    this->processingTime("measure");
    cout << "Skeletonization: "<< this->time*1000 << " ms" << endl;
}

void CropRowDetection::houghtTransform(){

    this->processingTime("start");

    //clear the previews line properties
    this->lines.clear();
    
    //creation of the lines vector, which will store the modules and angles of the lines in hough space
    // modules are stored in column 0 and the angles in column 1
    vector<Vec2f> hough_lines;

    //threshold is the minimum number of intercepts (curves intersecting at one point) to detect them
    HoughLines(this->images[5]->image, hough_lines, 1, CV_PI/60, 20, 0, 0); 
    
    // Draw the lines
    for( size_t i = 0; i < hough_lines.size(); i++ )
    {
        float rho = hough_lines[i][0], theta = hough_lines[i][1];
        Point p1, p2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        //cvRound is just a rounding function to transform double into an integer
        //calculates points p1 and p2 of type p = rho * cos (theta) + 1000 * sen (theta)
        p1.x = cvRound(x0 + 400*(-b));
        p1.y = cvRound(y0 + 400*(a));
        p2.x = cvRound(x0 - 400*(-b));
        p2.y = cvRound(y0 - 400*(a));
        //os pontos locais p1 e p2 calculados são gravados no método do objeto lines
        this->lines.push_back(new Line(p1,p2));
        //line desenha uma linha que conecta dois pontos (p1 e p2)
        line(this->images[6]->image, p1, p2, Scalar(255,80,100), 2, LINE_AA);
    }

    this->processingTime("measure");
    cout << "HoughTransform: "<< this->time*1000 << " ms" << endl;
}

void CropRowDetection::lineFilter(){

    this->processingTime("start");
    
    // the Filter checks whether the lines intersect. If they intersect -> do not plot
    
    //filteredLines is the line considered clean after the filter. So it stores the last filtered line
    vector <Line*>filteredLines;

    //Pushes the most voted line from HoughTransform to filteredLines
    filteredLines.push_back(this->lines[0]);
    
    //size_t is the unsigned integer type of the result of the sizeof
    for(size_t i=1; i < this->lines.size(); i++){

        int discart = 0;

        for(size_t n=0; n < filteredLines.size(); n++){
            
            Point interssectionP = this->intersectPoint(filteredLines[n]->pt1,filteredLines[n]->pt2, this->lines[i]->pt1, this->lines[i]->pt2);

            //If the returned point exists within the image, it means that there was an interception and does not plot
            //If p is out of the image or is discarded (-1, -1) then plots, because there was no interception
            //Count! = 0 means there was an interception
            if((interssectionP.x > 0) && (interssectionP.x < this->width) && (interssectionP.y > this->height/10) && (interssectionP.y < this->height)){
                discart++;
                break;
            }
        }

        //If count == 0, it means that the lines did not intersect, then plotting them
        if(discart == 0){

            //the current line of the loop is stored in filteredLines, considered a filtered line. This just doesn't happen in the 1st loop
            filteredLines.push_back(this->lines[i]);
            line(this->images[7]->image, this->lines[i]->pt1, this->lines[i]->pt2, Scalar(255,80,100), 2, LINE_AA);
        }
    }

    Rect rect(width/5, height/4, (3*width)/5, (3*height)/4);

    rectangle(this->images[0]->image, rect, Scalar(143, 131, 0), 2);

    this->processingTime("measure");

    cout << "Filtering Time : " << this->time*1000 << " ms" << endl << endl;
    cout << "Total Time : " << this->total_time*1000 << " ms" << endl;
    cout << "Expected for Real-time Processing: " << 0.0167*1000 << " ms" << endl;
    cout << "Real-time Margin: " << (0.0167 - this->total_time)*1000 << " ms" << endl << endl;
    cout << "Number of detected crop rows: " << filteredLines.size()-1 << endl << endl;

    this->start = 0;
    this->stop = 0;
    this->time = 0;
    this->total_time = 0;
}

Point CropRowDetection::intersectPoint(Point pt1, Point pt2, Point p3, Point p4){

    // calculate the coeficients for line 1 and 2
    // angular coef.
    float y1 = pt1.y - pt2.y; //delta y
    float x1 = pt2.x - pt1.x; //delta x

    // linear coef.
    float b1 = (pt1.x*pt2.y) - (pt2.x*pt1.y);

    float y2 = p3.y - p4.y;
    float x2 = p4.x - p3.x;
    float b2 = (p3.x*p4.y) - (p4.x*p3.y);

    //set the matrix a and b
    float a[2][2] = {{y1,x1},
                     {y2,x2}};
    float b[2][1] = {{b1},
                     {b2}};

    //create the angular and linear matrices
    Mat A = Mat(2,2, CV_32FC1, a);
    Mat B = Mat(2,1, CV_32FC1, b);

    float detA = determinant(A);

    //determinant
    // If determinant 0, they are the same line or are parallel, with LD vectors
    // If determinant! = 0, they intersect, with LI vectors
    if (detA != 0.00){
        //solve the eq ax=b
        //Then the point at which the interception occurs is calculated, which is returned by the function
        Mat x = A.inv() * (B*-1);
        return Point(x);
    }
    else{
        //discards the line by placing the intersection point outside the image
        return Point(-1,-1);
    }
}

void CropRowDetection::processingTime(string flag){

    if(flag == "start"){
        this->start = getTickCount();
    }
    else{
        this->stop = getTickCount();
        this->time = (this->stop - this->start)/ getTickFrequency();
        this->total_time = this->total_time + this->time;
    }

}

void CropRowDetection::compareImages(string sample_ExG, string sample_squeeze_unet , string path_to_ground_truth){

    Mat ExG = imread(sample_ExG, 0);
    Mat squeeze_unet = imread(sample_squeeze_unet, 0);
    Mat Ground_truth = imread(path_to_ground_truth, 0);

    threshold(Ground_truth, Ground_truth ,200 , 255, THRESH_BINARY);
    threshold(ExG, ExG ,200 , 255, THRESH_BINARY);

    if(ExG.empty() || squeeze_unet.empty() || Ground_truth.empty()){

        cerr << "Could not find the images, please check the path!" << endl;
        return;
    }
    
    int total_pixels_exg = ExG.cols*ExG.rows;
    float total_error_exg;
    float false_positives_exg = 0;
    float false_negatives_exg = 0;
    float center_error_exg = 0;
    float lost_samples_exg = 0;

    int total_pixels_squeeze_unet = squeeze_unet.cols*squeeze_unet.rows;
    float total_error_squeeze_unet;
    float false_positives_squeeze_unet = 0;
    float false_negatives_squeeze_unet = 0;
    float center_error_squeeze_unet = 0;
    float lost_samples_squeeze_unet = 0;
    float processed_pixels = 0;

    for(int y=0 ; y<ExG.rows; y++)
    {
        for(int x=0 ; x<ExG.cols ; x++)
        {           
            if((ExG.at<uchar>(x,y)) != Ground_truth.at<uchar>(x,y) && ((int)ExG.at<uchar>(x,y) > 200 && (int)Ground_truth.at<uchar>(x,y) < 200)){
                false_positives_exg++;
                if(y > ExG.rows/4 && x > ExG.cols/5 && x < (4*ExG.cols)/5){
                    center_error_exg++;
                } 
            }
            else if((ExG.at<uchar>(x,y)) != Ground_truth.at<uchar>(x,y) && ((int)ExG.at<uchar>(x,y) <= 200 && (int)Ground_truth.at<uchar>(x,y) >= 200)){
                false_negatives_exg++;
                if(y > ExG.rows/4 && x > ExG.cols/5 && x < (4*ExG.cols)/5){
                    center_error_exg++;
                } 
            }
            else{
                continue;
            }
            int z = (int)(Ground_truth.at<uchar>(x,y));
            cout << z << " ";
        }
    }

    for(int y=0 ; y<squeeze_unet.rows; y++)
    {
        for(int x=0 ; x<squeeze_unet.cols; x++)
        {           
            if((squeeze_unet.at<uchar>(x,y)) != Ground_truth.at<uchar>(x,y) && ((int)squeeze_unet.at<uchar>(x,y) > 200 && (int)Ground_truth.at<uchar>(x,y) < 200)){
                false_positives_squeeze_unet++;
                if(y > squeeze_unet.rows/4 && x > squeeze_unet.cols/5 && x < (4*squeeze_unet.cols)/5){
                    center_error_squeeze_unet++;
                } 
            }
            else if((squeeze_unet.at<uchar>(x,y)) != Ground_truth.at<uchar>(x,y) && ((int)squeeze_unet.at<uchar>(x,y) <= 200 && (int)Ground_truth.at<uchar>(x,y) >= 200)){
                false_negatives_squeeze_unet++;
                if(y > squeeze_unet.rows/4 && x > squeeze_unet.cols/5 && x < (4*squeeze_unet.cols)/5){
                    center_error_squeeze_unet++;
                } 
            }
            else{
                continue;
            }
        }
    }

    total_error_exg = false_positives_exg + false_negatives_exg;

    cout << "Processed pixels" << processed_pixels << endl;
    
    cout << "ExG results:" << endl;
    cout << endl << "Lost samples: " << (lost_samples_exg/total_pixels_exg)*100 << "%" << " of " << total_pixels_exg << " pixels" << endl << endl;
    cout << "From " << (1-(lost_samples_exg/total_pixels_exg))*100 << "% " << "(" << total_pixels_exg - lost_samples_exg << " pixels)" << " processed:" << endl;
    cout << "Pixel accuracy: " << (1-(total_error_exg/total_pixels_exg))*100 << "%" << endl;
    cout << "Pixel error: " << (total_error_exg/total_pixels_exg)*100 << "%"<< " , where:" << endl;
    cout << "> " << (false_positives_exg/total_error_exg)*100 << "% are False Positives and " << (false_negatives_exg/total_error_exg)*100 << "% are " << "False Negatives" << endl;
    cout << "> " << (center_error_exg/total_error_exg)*100 << "% are on central vision" << endl << endl;

    total_error_squeeze_unet = false_positives_squeeze_unet + false_negatives_squeeze_unet;
    
    cout << "squeeze_unet results:" << endl;
    cout << endl << "Lost samples: " << (lost_samples_squeeze_unet/total_pixels_squeeze_unet)*100 << "%" << " of " << total_pixels_squeeze_unet << " pixels" << endl << endl;
    cout << "From " << (1-(lost_samples_squeeze_unet/total_pixels_squeeze_unet))*100 << "% " << "(" << total_pixels_squeeze_unet - lost_samples_squeeze_unet << " pixels)" << " processed:" << endl;
    cout << "Pixel accuracy: " << (1-(total_error_squeeze_unet/total_pixels_squeeze_unet))*100 << "%" << endl;
    cout << "Pixel error: " << (total_error_squeeze_unet/total_pixels_squeeze_unet)*100 << "%"<< " , where:" << endl;
    cout << "> " << (false_positives_squeeze_unet/total_error_squeeze_unet)*100 << "% are False Positives and " << (false_negatives_squeeze_unet/total_error_squeeze_unet)*100 << "% are " << "False Negatives" << endl;
    cout << "> " << (center_error_squeeze_unet/total_error_squeeze_unet)*100 << "% are on central vision" << endl << endl;
}