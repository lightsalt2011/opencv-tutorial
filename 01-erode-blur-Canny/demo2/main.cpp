//
//  main.cpp
//  demo2
//
//  Created by lvfeng on 17/1/20.
//  Copyright © 2017年 lvfeng. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc, const char * argv[]) {
    // insert code here...
    
    std::cout << "demo 2 start\n";
    Mat srcImage = imread("/Users/lvfeng/trunk/opencv-tutorial/01-erode-blur-Canny/demo2/fruits.jpg");
    if (!srcImage.empty()) {
        imshow("demo 2", srcImage);
        waitKey(0);
    }
    std::cout <<" demo 2 end\n";
    
    
    std::cout<<"demo 3 start\n";
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat dstImage;
    erode(srcImage, dstImage, element);
    imshow("demo 3", dstImage);
    waitKey(0);
    std::cout <<" demo 3 end\n";
    
    std::cout<<"demo 4 start\n";
    blur(srcImage, dstImage, Size(7, 7));
    imshow("demo 4", dstImage);
    waitKey(0);
    std::cout<<"demo 4 end\n";

    std::cout<<"demo 5 start\n";
    Mat edge, grayImage;
    dstImage.create(srcImage.size(), srcImage.type()); //[1] create 与srcimage 同类型和大小的矩阵(dst)
    cvtColor(srcImage, grayImage, CV_BGR2GRAY);//[2] covert 原图到灰度图
    blur(grayImage, edge, Size(3, 3));//[3] 使用3*3 内核降噪
    Canny(edge, edge, 3, 9);
    imshow("demo 5", edge);
    waitKey(0);
    std::cout<<"demo 5 end\n";
/*
    std::cout<<"demo 6 start\n";
    VideoCapture capture(0);
    while(1) {
        Mat frame; //定义Mat变量，用于存储每一帧的图像
        capture>>frame;
        imshow("demo6", frame);
        waitKey(30);
    }
    std::cout<<"demo 6 end\n";
*/
    std::cout<<"demo 7 start\n";
    VideoCapture capture(0);
    Mat edges;
    while(1) {
        Mat frame;
        capture>>frame;
        cvtColor(frame, edges, CV_BGR2GRAY);//转换BGR彩色图到灰度图
        blur(frame, edges, Size(10, 10));//进行模糊
        Canny(edges, edges, 0, 30);//进行Canny 边缘检测并显示
        imshow("demo 7", edges);
        if (waitKey(30) >= 0)
            break;
    }
    
    
    
    
    return 0;
}
