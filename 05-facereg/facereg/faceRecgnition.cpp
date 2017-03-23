
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "detect_recog.h"

using namespace std;
using namespace cv;

CvMemStorage* storage = 0;
CvHaarClassifierCascade* cascade = 0;
CvHaarClassifierCascade* nested_cascade = 0;
int use_nested_cascade = 0;
const char* cascade_name =
    "./data/haarcascade_frontalface_alt.xml";//±»À“—æ≠—µ¡∑∫√µƒ»À¡≥ºÏ≤‚xml ˝æ›
const char* nested_cascade_name =
    "./data/haarcascade_eye_tree_eyeglasses.xml";
VideoCapture capture;
IplImage *frame, *frame_copy = 0;
IplImage *image = 0;
const char* scale_opt = "--scale="; // ∑÷¿‡∆˜—°œÓ÷∏ æ∑˚∫≈ 
int scale_opt_len = (int)strlen(scale_opt);
const char* cascade_opt = "--cascade=";
int cascade_opt_len = (int)strlen(cascade_opt);
const char* nested_cascade_opt = "--nested-cascade";
int nested_cascade_opt_len = (int)strlen(nested_cascade_opt);
double scale = 1;
int num_components = 9;
double facethreshold = 9.0;
//opencvµƒFaceRecogizerƒø«∞”–»˝∏ˆ¿‡ µœ÷¡ÀÀ˚£¨Ãÿ’˜¡≥∫Õfisherface∑Ω∑®◊Ó…Ÿ—µ¡∑ÕºœÒŒ™¡Ω’≈£¨∂¯LBPø…“‘µ•’≈ÕºœÒ—µ¡∑
//cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer();
//cv::Ptr<cv::FaceRecognizer> model = cv::createFisherFaceRecognizer();
cv::Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer();//LBPµƒ’‚∏ˆ∑Ω∑®‘⁄µ•∏ˆ»À¡≥—È÷§∑Ω√Ê–ßπ˚◊Ó∫√

vector<Mat> images;//¡Ω∏ˆ»›∆˜images,labels¿¥¥Ê∑≈ÕºœÒ ˝æ›∫Õ∂‘”¶µƒ±Í«©
vector<int> labels;


int main( int argc, char** argv )
{
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0); //º”‘ÿ∑÷¿‡∆˜ 
    if(!cascade) 
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
		getchar();
        return -1;
    }
	model->set("threshold", 2100.0);
	string output_folder;
	output_folder = string("./einfacedata");

	//∂¡»°ƒ„µƒCSVŒƒº˛¬∑æ∂
	string fn_csv = string("./einfacedata/at.txt");
	try
	{
		//Õ®π˝./einfacedata/at.txt’‚∏ˆŒƒº˛∂¡»°¿Ô√Êµƒ—µ¡∑ÕºœÒ∫Õ¿‡±±Í«©
		read_csv(fn_csv, images, labels);	
	}
	catch(cv::Exception &e)
	{
		cerr<<"Error opening file "<<fn_csv<<". Reason: "<<e.msg<<endl;
		exit(1);
	}
	/*
	//read_img’‚∏ˆ∫Ø ˝÷±Ω”¥”einfacedata/trainingdataƒø¬ºœ¬∂¡»°ÕºœÒ ˝æ›≤¢ƒ¨»œΩ´ÕºœÒ÷√Œ™0
	//À˘“‘»Áπ˚”√’‚∏ˆ∫Ø ˝÷ªƒ‹”√¿¥µ•∏ˆ»À¡≥—È÷§
	if(!read_img(images, labels))
	{
		cout<< "Error in reading images!";
		images.clear();
		labels.clear();
		return 0;
	}
	*/
	cout << images.size() << ":" << labels.size()<<endl;
	//»Áπ˚√ª”–∂¡µΩ◊„πªµƒÕº∆¨£¨æÕÕÀ≥ˆ
	if(images.size() <= 2)
	{
		string error_message = "This demo needs at least 2 images to work.";
		CV_Error(CV_StsError, error_message);
	}

	//µ√µΩµ⁄“ª’≈’’∆¨µƒ∏ﬂ∂»£¨‘⁄œ¬√Ê∂‘ÕºœÒ±‰–ŒµΩÀ˚√«‘≠ º¥Û–° ±–Ë“™
	//int height = images[0].rows;
	//“∆≥˝◊Ó∫Û“ª’≈Õº∆¨£¨”√”⁄◊ˆ≤‚ ‘
	//Mat testSample = images[images.size() - 1];
	//cv::imshow("testSample", testSample);
	//int testLabel = labels[labels.size() - 1];
	//images.pop_back();
	//labels.pop_back();

	//œ¬√Ê¥¥Ω®“ª∏ˆÃÿ’˜¡≥ƒ£–Õ”√”⁄»À¡≥ ∂±£¨
	// Õ®π˝CSVŒƒº˛∂¡»°µƒÕºœÒ∫Õ±Í«©—µ¡∑À¸°£

	//Ω¯––—µ¡∑
	model->train(images, labels);

    storage = cvCreateMemStorage(0); // ¥¥Ω®ƒ⁄¥Ê¥Ê¥¢∆˜   
    //capture = cvCaptureFromCAM(0); // ¥¥Ω® ”∆µ∂¡»°Ω·ππ
    cvNamedWindow( "result", 1 );
    capture.open( 0 );
    
    Mat I;
    //IplImage* pI     = &I.operator IplImage();
    //Ptr<IplImage> piI = &I.operator IplImage();


    
    if( capture.isOpened()  ) // »Áπ˝ « ”∆µªÚ…„œÒÕ∑≤…ºØÕºœÒ£¨‘Ú—≠ª∑¥¶¿Ì√ø“ª÷°
    {
        for(;;)
        {
            capture >> I;
            
            IplImage Iframe = IplImage(I);
            frame = &Iframe;
            
            if( !frame_copy )
                //frame_copy = cvCreateImage( cvSize(640,480),IPL_DEPTH_8U, frame->nChannels );
                frame_copy = cvCreateImage( cvSize(frame->width, frame->height), 8, frame->nChannels );
           
            if( frame->origin == IPL_ORIGIN_TL )
                cvCopy( frame, frame_copy, 0 );
            else
                cvFlip( frame, frame_copy, 0 );
           
            //detect_and_draw( frame ); // »Áπ˚µ˜”√’‚∏ˆ∫Ø ˝£¨÷ª « µœ÷»À¡≥ºÏ≤‚
			cout << frame_copy->width << "x" << frame_copy->height << endl;
			recog_and_draw( frame_copy );//∏√∫Ø ˝ µœ÷»À¡≥ºÏ≤‚∫Õ ∂±
            if( cvWaitKey( 100 ) >= 0 )//escº¸÷µ∫√œÒ «100
                goto _cleanup_;
            
        }
        
    _cleanup_:
        cvReleaseImage( &frame_copy );
        //cvReleaseCapture( &capture );
    }    
    cvDestroyWindow("result");
    return 0;
}

