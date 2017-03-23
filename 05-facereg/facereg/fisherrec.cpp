#include "stdafx.h"/*


#include "cv.h"
#include "highgui.h"

#include <opencv2\contrib\contrib.hpp>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp> 

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

static  Mat norm_0_255(cv::InputArray _src)
{
	Mat src = _src.getMat();
	Mat dst;

	switch(src.channels())
	{
	case 1:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}

	return dst;
}

static void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';')
{
	std::ifstream file(filename.c_str(), ifstream::in);
	if(!file)
	{
		string error_message = "No valid input file was given.";
		CV_Error(CV_StsBadArg, error_message);
	}

	string line, path, classlabel;
	while(getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);  //�����ֺžͽ���
		getline(liness, classlabel);     //�����ӷֺź��濪ʼ���������н���
		if(!path.empty() && !classlabel.empty())
		{
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, char *argv[])
{
	string output_folder;
	output_folder = string("F:/VS2010/faceRecgnition/einfacedata");

	//��ȡ���CSV�ļ�·��
	string fn_csv = string("F:/VS2010/faceRecgnition/einfacedata/at.txt");

	//�������������ͼ�����ݺͶ�Ӧ�ı�ǩ
	vector<Mat> images;
	vector<int> labels;

	try
	{
		read_csv(fn_csv, images, labels);	
	}
	catch(cv::Exception &e)
	{
		cerr<<"Error opening file "<<fn_csv<<". Reason: "<<e.msg<<endl;
		exit(1);
	}

	//���û�ж����㹻��ͼƬ�����˳�
	if(images.size() <= 1)
	{
		string error_message = "This demo needs at least 2 images to work.";
		CV_Error(CV_StsError, error_message);
	}

	//�õ���һ����Ƭ�ĸ߶ȣ��������ͼ����ε�����ԭʼ��Сʱ��Ҫ
	int height = images[0].rows;

	//�Ƴ����һ��ͼƬ������������
	Mat testSample = images[images.size() - 1];
	cv::imshow("testSample", testSample);
	int testLabel = labels[labels.size() - 1];

	images.pop_back();
	labels.pop_back();




	cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer();

	//training
	model->train(images, labels);

	
	//����Բ���ͼ�����Ԥ�⣬predictedLabel ��Ԥ���ǩ���
	int predictedLabel = model->predict(testSample);

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout<<result_message<<endl;


	//��ȡ������ģ�͵�����ֵ�����ӣ�ʹ����getMat����
	Mat eigenvalues = model->getMat("eigenvalues");
	//��ȡ��������
	Mat W = model->getMat("eigenvectors");

	//�õ�ѵ��ͼ��ľ�ֵ����
	Mat mean = model->getMat("mean");

	imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	cv::imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

	//ʵ�ֲ�����������
	for(int i=0; i <min(10, W.cols); i++)
	{
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout<<msg<<endl;

		Mat ev = W.col(i).clone();

		//�������ԭʼ��С��Ϊ�˰�������ʾ��һ����0-255.
		Mat grayscale = norm_0_255(ev.reshape(1,height));

		//ʹ��α��ɫ����ʾ���
		Mat cgrayscale;
		cv::applyColorMap(grayscale, cgrayscale, COLORMAP_JET);

		imshow(format("eigenface_%d", i), cgrayscale);
		imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), cgrayscale);
	}
	
	//��Ԥ������У���ʾ�������ؽ����ͼƬ
	for(int num_components = 10; num_components < 390; num_components += 15)
	{
		//��ģ���е�����������ȡһ����
		Mat evs = Mat(W, Range::all(), Range(0, num_components));
		//ͶӰ
		Mat projection = cv::subspaceProject(evs, mean, images[0].reshape(1,1));
		//�ع�
		Mat reconstruction = cv::subspaceReconstruct(evs, mean, projection);

		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

		imshow(format("eigenface_reconstruction_%d", num_components),reconstruction);
		imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(),num_components), reconstruction);

	}
	
	cv::waitKey(0);
	return 0;
}
*/