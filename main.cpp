/*
author by D mortem
Harris Corner Detector
*/

#include "cv.h"
#include "highgui.h"
#include <opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <math.h>
using namespace cv;
double pi = 3.1415926535898;

void gass(cv::Mat &gas, int o, int Size)
{
	double tmp, k;
	for (int i = -Size / 2; i <= Size / 2; i++)
		for (int j = -Size / 2; j <= Size / 2; j++)
		{
			k = -((i * i + j * j)*1.0 / (2.0 * o * o));
			tmp = exp(k);
			gas.at<float>(i + Size / 2, j + Size / 2) = tmp / (2 * pi * o * o);
			//printf("%lf  ", gas.at<float>(i + Size / 2, j + Size / 2));
		}
}

void expand_Img_input(const cv::Mat &Img_input, cv::Mat &Img_expand)
{
	// ԭͼ������չ���ͼ�����룬�ճ���0��/�к͵�Img_input.size().height+1��/��
	cv::Mat tmp = Img_expand(cv::Range(1, Img_input.size().height + 1), cv::Range(1, Img_input.size().width + 1));
	Img_input.copyTo(tmp);

	// ԭͼ�������ҵ��ı߷ֱ�ָ��Ƶ���ͼ���ıߣ������ĸ���ֻ�ܿճ���
	tmp = Img_expand(cv::Range(0, 1), cv::Range(1, Img_input.size().width + 1));
	Img_input.row(0).copyTo(tmp);
	tmp = Img_expand(cv::Range(Img_input.size().height + 1, Img_input.size().height + 2), cv::Range(1, Img_input.size().width + 1));
	Img_input.row(Img_input.size().height - 1).copyTo(tmp);
	tmp = Img_expand(cv::Range(1, Img_input.size().height + 1), cv::Range(0, 1));
	Img_input.col(0).copyTo(tmp);
	tmp = Img_expand(cv::Range(1, Img_input.size().height + 1), cv::Range(Img_input.size().width + 1, Img_input.size().width + 2));
	Img_input.col(Img_input.size().width - 1).copyTo(tmp);

	// ��Img_expandת��Ϊ�Ҷ�ͼ�񣬿���ֱ����I��ʾ���������������������ھ�����
	cv::cvtColor(Img_expand, Img_expand, cv::COLOR_BGR2GRAY);
}

// ����
float convolution(cv::Mat &tmp, const cv::Mat &g, int x, int y)	// gΪ�������tmpΪԭ����
{
	int size = g.size().height;			// �õ���������С
	float sum = 0;
	for (int i = 0; i < size; i++)		// ���ݾ�������С���д���
		for (int j = 0; j < size; j++)
			sum += g.at<float>(i, j) * tmp.at<float>(x + i, y + j);
	return sum;
}

// �����ݶȵĺ���
void calculate_gradient(cv::Mat &Img_gradientX, cv::Mat &Img_gradientY, cv::Mat &Img_expand)
{
	// ����sobel���ӵľ��󣬴��ڴ�СΪ3
	const cv::Mat sobelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	const cv::Mat sobelY = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	//cv::Mat tmp;
	int height = Img_gradientX.size().height;
	int width = Img_gradientX.size().width;
	//��������ɨ��ͼƬ
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++) 
		{
			Img_gradientX.at<float>(i, j) = convolution(Img_expand, sobelX, i, j);	//Img_gradientX[i, j]��Img_expand[i+1, j+1]��x������ݶȡ�i��j��¼��ǰλ��
			Img_gradientY.at<float>(i, j) = convolution(Img_expand, sobelY, i, j);
		}
	}
}
void pick_R_local_max(cv::Mat &R, double threshold)
{
	//	��window*window�Ĵ��ڲ��Ҿֲ����ֵ�������Ǿֲ����ֵ��Ҳ������ֵthreshold�ĵ㣬R = 0
	double maxVal = 0, tmp;
	int windowSize = 30;
	int m = R.size().height, n = R.size().width;
	for (int i = 0; i < m - windowSize; i += windowSize) 
	{
		for (int j = 0; j < n - windowSize; j += windowSize) 
		{
			maxVal = 0;
			for (int k = i; k < i + windowSize; ++k) 
			{
				for (int l = j; l < j + windowSize; ++l) 
				{
					tmp = R.at<float>(k, l);
					if (tmp - maxVal > 1e-7 && tmp - threshold > 1e-7) 
					{
						maxVal = R.at<float>(k, l);
					}
					else {
						R.at<float>(k, l) = 0.0f;
					}
				}
			}
		}
	}

	// ������1��
	for (int k = 0; k < n - windowSize; k += windowSize) 
	{
		maxVal = 0;
		for (int i = m - windowSize; i < m; ++i) 
		{
			for (int j = k; j < k + windowSize; ++j) 
			{
				tmp = R.at<float>(i, j);
				if (tmp - maxVal > 1e-7 && tmp - threshold > 1e-7) 
				{
					maxVal = R.at<float>(i, j);
				}
				else {
					R.at<float>(i, j) = 0.0f;
				}
			}
		}
	}
	// ���ұ�1��
	for (int k = 0; k < m - windowSize; k += windowSize) 
	{
		maxVal = 0;
		for (int i = k; i < k + windowSize; ++i) 
		{
			for (int j = n - windowSize; j < n; ++j) 
			{
				tmp = R.at<float>(i, j);
				if (tmp - maxVal > 1e-7 && tmp - threshold > 1e-7) 
				{
					maxVal = R.at<float>(i, j);
				}
				else {
					R.at<float>(i, j) = 0.0f;
				}
			}
		}
	}
	// ���½�
	maxVal = 0;
	for (int i = m - windowSize; i < m; ++i) 
	{
		for (int j = n - windowSize; j < n; ++j) 
		{
			tmp = R.at<float>(i, j);
			if (tmp - maxVal > 1e-7 && tmp - threshold > 1e-7) 
			{
				maxVal = R.at<float>(i, j);
			}
			else {
				R.at<float>(i, j) = 0.0f;
			}
		}
	}
}

void draw_corner(cv::Mat &R, cv::Mat &Img_input, int windowSize)
{
	//���ҵ��ǵ��λ�û�Ȧ����ע��λ��
	int m = R.size().height, n = R.size().width;
	for (int i = windowSize / 2, k = 0; k < m; ++i, ++k) 
	{
		for (int j = windowSize / 2, l = 0; l < n; ++j, ++l) 
		{
			if (R.at<float>(k, l) > 1e-7)	// �����Ǽ���ֵ���ƺ󣬱�֤�˳��ǵ���ĵ��Rֵ��Ϊ0
			{
				cv::circle(Img_input, cvPoint(j, i), 1, CV_RGB(255, 0, 0));
				//printf("%.2f  ", R.at<float>(k, l));
			}
		}
	}
}

void calculate_M(cv::Mat &Img_gradientX, cv::Mat &Img_gradientY, cv::Mat &M, int row, int col, int windowSize)
{
	//���չ�ʽ���� M�����Ix^2��Iy^2��Ix*Iy������w(x,y)ѡ��"1 in window, 0 out of window"���ִ��ں���
	for (int i = row - windowSize / 2; i < row + windowSize / 2 + 1; ++i) 
	{
		for (int j = col - windowSize / 2; j < col + windowSize / 2 + 1; ++j) 
		{
			M.at<float>(0, 0) += Img_gradientX.at<float>(i, j) * Img_gradientX.at<float>(i, j);
			M.at<float>(1, 0) += Img_gradientX.at<float>(i, j) * Img_gradientY.at<float>(i, j);
			M.at<float>(0, 1) += Img_gradientX.at<float>(i, j) * Img_gradientY.at<float>(i, j);
			M.at<float>(1, 1) += Img_gradientY.at<float>(i, j) * Img_gradientY.at<float>(i, j);
		}
	}
}

void calculate_lambda(cv::Mat &M, float &lambda1, float &lambda2)
{
	//����lambda��ֵ. assign M = [a x] = [a x		] = [a(ab - x^2)/x^2  0			  ]
	//							  x b	  0 ab/x - x	 0				  (ab - x^2)/x
	//float a = M.at<float>(0, 0), b = M.at<float>(1, 1), x = M.at<float>(0, 1);
	//lambda1 = a * a / x * b / x - a;
	//lambda2 = a * b / x - x;

	float a = 1.0f, b = -cv::trace(M).val[0], c = cv::determinant(M);	// ���ݽ��۷�������lambda
	lambda1 = 0.5f / a * (-b + sqrt(b*b - 4 * a * c));
	lambda2 = 0.5f / a * (-b - sqrt(b*b - 4 * a * c));
	//printf("%.6lf %.6lf", lambda1, lambda2);
}

void normalize_to_Range_0_255(cv::Mat &M)
{
	//��ͼ���һ����ʹ���䴦�������ĻҶ�ֵ��Χ��
	double minVal = 0.0, maxVal = 255.0;
	cv::minMaxIdx(M, &minVal, &maxVal);		// ����M�����е����ֵ��Сֵ
	M = (M - minVal) / (maxVal - minVal);
}

void calculate_R(cv::Mat &Img_expand, cv::Mat &R, const cv::Size &R_Size, float K, int windowSize, double threshold)
{
	cv::Mat Min_lambda(R_Size, CV_32F);				// �����������
	cv::Mat Max_lambda(R_Size, CV_32F);				// ��С��������
	float lambda1, lambda2;							// lambda1Ϊ�������ֵ��lambda2Ϊ��С����ֵ
	cv::Mat Img_gradientX(Img_expand.size().height - 2, Img_expand.size().width - 2, CV_32F);	// x������ݶȣ�����Img_gradientX[0,0]��Img_expand[1,1]���ݶ�
	cv::Mat Img_gradientY(Img_expand.size().height - 2, Img_expand.size().width - 2, CV_32F);	// y������ݶ�

	calculate_gradient(Img_gradientX, Img_gradientY, Img_expand);	// 1������ͼ���ݶȣ����Ix��Iy

	int m = R_Size.height, n = R_Size.width;
	// k,l��res��������ꣻi,j��gre��������ꡣ��ΪgreҪ��res�����(k,l)��(i,j)�������غϣ�ֻ����������ϵ��ͬ��������겻ͬ
	for (int i = windowSize / 2, k = 0; k < m; ++i, ++k) 
	{
		for (int j = windowSize / 2, l = 0; l < n; ++j, ++l) 
		{
			cv::Mat M(2, 2, CV_32F, cv::Scalar(0.0));
			calculate_M(Img_gradientX, Img_gradientY, M, i, j, windowSize);	// 2������õ���(i,j)��Ӧ��M����ÿ���㶼��1��M����M����Ҫͨ��window_Size*window_Size�ľ�����Ϣͨ��һ������õ�
			calculate_lambda(M, lambda1, lambda2);							// 3������M�������(i,j)���2������ֵ
			Max_lambda.at<float>(k, l) = sqrt(sqrt(lambda1));							
			Min_lambda.at<float>(k, l) = sqrt(sqrt(lambda2));							
												
			R.at<float>(k, l) = sqrt(sqrt(lambda1 * lambda2 - K * (lambda1 + lambda2) * (lambda1 + lambda2)));	 // 4������Rֵ
		}
	}

	normalize_to_Range_0_255(Min_lambda);				// ��һ��
	cv::namedWindow("Min_lambda");
	cv::imshow("Min_lambda", Min_lambda);				// ����155Ч������ȥ����
	cv::imwrite("Min_lambda.jpg", Min_lambda * 255);	// imshow��ȡ�ľ������ݷ�ΧΪ[0,1]����imwrite��ȡ�����ݷ�ΧΪ[0,255]

	normalize_to_Range_0_255(Max_lambda);
	cv::namedWindow("Max_lambda");
	cv::imshow("Max_lambda", Max_lambda);
	cv::imwrite("Max_lambda.jpg", Max_lambda * 255);

	//cv::cvtColor(R, R, cv::COLOR_GRAY2BGR);
	cv::threshold(R, R, 0, 0, cv::THRESH_TOZERO);		// ���R<0, ʹ��R=0��ʹflat��edge��Rֵ��ΪС������ֻ��corner�Ǵ������������Ļ�Ч�����ƣ�ֻ������ֵ����ȷ�����ұ���(flat)����
	normalize_to_Range_0_255(R);
	cv::namedWindow("R");
	cv::imshow("R", R);
	cv::imwrite("R.jpg", R * 255);
}

//template <typename T>
void getColorR(Mat &src, Mat &res)
{
	int row = src.size().height;
	int col = src.size().width;
	float tmp1 = 0;
	float tmp2 = 0;
	//������ɫRͼ
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			tmp1 = src.at<float>(i, j) * 255;
			if (tmp1 > tmp2)
				tmp2 = tmp1;
			if (tmp1 <= 51)
			{
				res.at<cv::Vec3f>(i, j)[0] = 255;
				res.at<cv::Vec3f>(i, j)[1] = tmp1 * 5;
				res.at<cv::Vec3f>(i, j)[2] = 0;
			}
			else if (tmp1 <= 102)
			{
				tmp1 -= 51;
				res.at<cv::Vec3f>(i, j)[0] = 255 - tmp1 * 5;
				res.at<cv::Vec3f>(i, j)[1] = 255;
				res.at<cv::Vec3f>(i, j)[2] = 0;
			}
			else if (tmp1 <= 153)
			{
				tmp1 -= 102;
				res.at<cv::Vec3f>(i, j)[0] = 0;
				res.at<cv::Vec3f>(i, j)[1] = 255;
				res.at<cv::Vec3f>(i, j)[2] = tmp1 * 5;
			}
			else if (tmp1 <= 204)
			{
				tmp1 -= 153;
				res.at<cv::Vec3f>(i, j)[0] = 0;
				res.at<cv::Vec3f>(i, j)[1] = 255 - uchar(128.0*tmp1 / 51.0 + 0.5);
				res.at<cv::Vec3f>(i, j)[2] = 255;
			}
			else
			{
				tmp1 -= 204;
				res.at<cv::Vec3f>(i, j)[0] = 0;
				res.at<cv::Vec3f>(i, j)[1] = 127 - uchar(127.0*tmp1 / 51.0 + 0.5);
				res.at<cv::Vec3f>(i, j)[2] = 255;
			}
		}
	}
	//printf("%.2f\n", tmp2);
}

int main(int argc, char *argv[])
{
// ��ʼ��
	float k = atof(argv[2]);			// ����Rʱ�ľ������k
	int windowSize = atof(argv[3]);		// ���ں���w(x,y)�Ĵ��ڴ�С
	double threshold = sqrt(sqrt(0.28/255));			// ��ֵ

	cv::Mat Img_input = cv::imread(argv[1]);		// ����ͼƬ

	// ����һ��Img_expand����������ͨ��BGR����ֵ��Ϊ0
	cv::Mat Img_expand(Img_input.size().height + 2, Img_input.size().width + 2, CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));	
	cv::Size R_Size(Img_input.size().width - windowSize / 2 * 2, Img_input.size().height - windowSize / 2 * 2);
	cv::Mat R(R_Size, CV_32F);			// ����Rֵ����

	expand_Img_input(Img_input, Img_expand);		// Ϊ��ʹ��Img_input������ÿһ���㶼����������ݶȣ���Ҫ�������������2��2��
	Img_expand.convertTo(Img_expand, CV_32F);		// Imread��������unsigned char���ͣ���Ҫת����float���ڼ��㣬���پ�����ʧ
	//����Rֵ�����С��С��gre�ݶȺ�������Ϊ����R��(i,j)����ݶȣ�����Sobel����Ҫȥ����(i-window/2,j-window/2)~(i+window/2,j+window/2)λ�õ�I

// ��˹�˲�
	//GaussianBlur(Img_expand, Img_expand, cv::Size(5, 5), 1, 1);
	cv::Mat gas(5, 5, CV_32F);	// �����˹����
	gass(gas, 1, 5);
	for (int i = 0; i < Img_expand.size().height-4; i++)
	{
		for (int j = 0; j < Img_expand.size().width-4; j++)
		{
			Img_expand.at<float>(i, j) = convolution(Img_expand, gas, i, j);	//Img_gradientX[i, j]��Img_expand[i+1, j+1]��x������ݶȡ�i��j��¼��ǰλ��
		}
	}

// ����Rֵ							
	calculate_R(Img_expand, R, R_Size, k, windowSize, threshold);	

// �����ɫRͼ
	cv::Mat R1(R_Size, CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
	getColorR(R, R1);
	cv::imshow("ColorR", R1);				
	cv::imwrite("ColorR.jpg", R1 * 255);

// �Ǽ���ֵ���ƣ��õ�ѡȡ��Ľǵ�
	pick_R_local_max(R, threshold);

// ���ǵ��Ժ�ɫ�ĵ����ʽ������ԭͼ��
	draw_corner(R, Img_input, windowSize);			// ��ԭͼ�ϻ��ƽǵ㣬windowSize��������ԭͼImg_input��R��������֮���������ϵ��Img_input�϶�Ӧ�������Ҫ��R�����windowSize/2
	cv::imshow("result", Img_input);				// ����Img_input�������ͨ��BGRֵ�����ֱ���������
	cv::imwrite("result.jpg", Img_input);
	cv::waitKey();

	return 0;
}