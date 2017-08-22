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
	// 原图放在扩展后的图像中央，空出第0行/列和第Img_input.size().height+1行/列
	cv::Mat tmp = Img_expand(cv::Range(1, Img_input.size().height + 1), cv::Range(1, Img_input.size().width + 1));
	Img_input.copyTo(tmp);

	// 原图上下左右的四边分别分复制到新图的四边，不过四个角只能空出来
	tmp = Img_expand(cv::Range(0, 1), cv::Range(1, Img_input.size().width + 1));
	Img_input.row(0).copyTo(tmp);
	tmp = Img_expand(cv::Range(Img_input.size().height + 1, Img_input.size().height + 2), cv::Range(1, Img_input.size().width + 1));
	Img_input.row(Img_input.size().height - 1).copyTo(tmp);
	tmp = Img_expand(cv::Range(1, Img_input.size().height + 1), cv::Range(0, 1));
	Img_input.col(0).copyTo(tmp);
	tmp = Img_expand(cv::Range(1, Img_input.size().height + 1), cv::Range(Img_input.size().width + 1, Img_input.size().width + 2));
	Img_input.col(Img_input.size().width - 1).copyTo(tmp);

	// 将Img_expand转化为灰度图像，可以直接用I表示其像素特征，便于做后期矩阵处理
	cv::cvtColor(Img_expand, Img_expand, cv::COLOR_BGR2GRAY);
}

// 算卷积
float convolution(cv::Mat &tmp, const cv::Mat &g, int x, int y)	// g为卷积矩阵，tmp为原矩阵
{
	int size = g.size().height;			// 得到卷积矩阵大小
	float sum = 0;
	for (int i = 0; i < size; i++)		// 根据卷积矩阵大小进行处理
		for (int j = 0; j < size; j++)
			sum += g.at<float>(i, j) * tmp.at<float>(x + i, y + j);
	return sum;
}

// 计算梯度的函数
void calculate_gradient(cv::Mat &Img_gradientX, cv::Mat &Img_gradientY, cv::Mat &Img_expand)
{
	// 定义sobel算子的矩阵，窗口大小为3
	const cv::Mat sobelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	const cv::Mat sobelY = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	//cv::Mat tmp;
	int height = Img_gradientX.size().height;
	int width = Img_gradientX.size().width;
	//逐行逐列扫过图片
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++) 
		{
			Img_gradientX.at<float>(i, j) = convolution(Img_expand, sobelX, i, j);	//Img_gradientX[i, j]是Img_expand[i+1, j+1]在x方向的梯度。i与j记录当前位置
			Img_gradientY.at<float>(i, j) = convolution(Img_expand, sobelY, i, j);
		}
	}
}
void pick_R_local_max(cv::Mat &R, double threshold)
{
	//	在window*window的窗口查找局部最大值，若不是局部最大值但也超过阈值threshold的点，R = 0
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

	// 最下面1行
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
	// 最右边1列
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
	// 右下角
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
	//在找到角点的位置画圈，标注其位置
	int m = R.size().height, n = R.size().width;
	for (int i = windowSize / 2, k = 0; k < m; ++i, ++k) 
	{
		for (int j = windowSize / 2, l = 0; l < n; ++j, ++l) 
		{
			if (R.at<float>(k, l) > 1e-7)	// 做过非极大值抑制后，保证了除角点外的点的R值均为0
			{
				cv::circle(Img_input, cvPoint(j, i), 1, CV_RGB(255, 0, 0));
				//printf("%.2f  ", R.at<float>(k, l));
			}
		}
	}
}

void calculate_M(cv::Mat &Img_gradientX, cv::Mat &Img_gradientY, cv::Mat &M, int row, int col, int windowSize)
{
	//按照公式计算 M，求出Ix^2、Iy^2、Ix*Iy，其中w(x,y)选择"1 in window, 0 out of window"那种窗口函数
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
	//计算lambda的值. assign M = [a x] = [a x		] = [a(ab - x^2)/x^2  0			  ]
	//							  x b	  0 ab/x - x	 0				  (ab - x^2)/x
	//float a = M.at<float>(0, 0), b = M.at<float>(1, 1), x = M.at<float>(0, 1);
	//lambda1 = a * a / x * b / x - a;
	//lambda2 = a * b / x - x;

	float a = 1.0f, b = -cv::trace(M).val[0], c = cv::determinant(M);	// 根据结论反过来求lambda
	lambda1 = 0.5f / a * (-b + sqrt(b*b - 4 * a * c));
	lambda2 = 0.5f / a * (-b - sqrt(b*b - 4 * a * c));
	//printf("%.6lf %.6lf", lambda1, lambda2);
}

void normalize_to_Range_0_255(cv::Mat &M)
{
	//将图像归一化，使得其处于正常的灰度值范围内
	double minVal = 0.0, maxVal = 255.0;
	cv::minMaxIdx(M, &minVal, &maxVal);		// 返回M矩阵中的最大值最小值
	M = (M - minVal) / (maxVal - minVal);
}

void calculate_R(cv::Mat &Img_expand, cv::Mat &R, const cv::Size &R_Size, float K, int windowSize, double threshold)
{
	cv::Mat Min_lambda(R_Size, CV_32F);				// 最大特征矩阵
	cv::Mat Max_lambda(R_Size, CV_32F);				// 最小特征矩阵
	float lambda1, lambda2;							// lambda1为最大特征值，lambda2为最小特征值
	cv::Mat Img_gradientX(Img_expand.size().height - 2, Img_expand.size().width - 2, CV_32F);	// x方向的梯度，其中Img_gradientX[0,0]是Img_expand[1,1]的梯度
	cv::Mat Img_gradientY(Img_expand.size().height - 2, Img_expand.size().width - 2, CV_32F);	// y方向的梯度

	calculate_gradient(Img_gradientX, Img_gradientY, Img_expand);	// 1、计算图像梯度，求出Ix和Iy

	int m = R_Size.height, n = R_Size.width;
	// k,l是res矩阵的坐标；i,j是gre矩阵的坐标。因为gre要比res大，因此(k,l)和(i,j)两点是重合，只是由于坐标系不同，因此坐标不同
	for (int i = windowSize / 2, k = 0; k < m; ++i, ++k) 
	{
		for (int j = windowSize / 2, l = 0; l < n; ++j, ++l) 
		{
			cv::Mat M(2, 2, CV_32F, cv::Scalar(0.0));
			calculate_M(Img_gradientX, Img_gradientY, M, i, j, windowSize);	// 2、计算得到点(i,j)对应的M矩阵。每个点都有1个M矩阵，M矩阵要通过window_Size*window_Size的矩阵信息通过一定计算得到
			calculate_lambda(M, lambda1, lambda2);							// 3、根据M矩阵计算(i,j)点的2个特征值
			Max_lambda.at<float>(k, l) = sqrt(sqrt(lambda1));							
			Min_lambda.at<float>(k, l) = sqrt(sqrt(lambda2));							
												
			R.at<float>(k, l) = sqrt(sqrt(lambda1 * lambda2 - K * (lambda1 + lambda2) * (lambda1 + lambda2)));	 // 4、计算R值
		}
	}

	normalize_to_Range_0_255(Min_lambda);				// 归一化
	cv::namedWindow("Min_lambda");
	cv::imshow("Min_lambda", Min_lambda);				// 除以155效果看上去更好
	cv::imwrite("Min_lambda.jpg", Min_lambda * 255);	// imshow读取的矩阵数据范围为[0,1]，而imwrite读取的数据范围为[0,255]

	normalize_to_Range_0_255(Max_lambda);
	cv::namedWindow("Max_lambda");
	cv::imshow("Max_lambda", Max_lambda);
	cv::imwrite("Max_lambda.jpg", Max_lambda * 255);

	//cv::cvtColor(R, R, cv::COLOR_GRAY2BGR);
	cv::threshold(R, R, 0, 0, cv::THRESH_TOZERO);		// 如果R<0, 使得R=0，使flat和edge的R值都为小正数，只有corner是大正数。不做的话效果类似，只不过阈值较难确定，且背景(flat)会变灰
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
	//构建彩色R图
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
// 初始化
	float k = atof(argv[2]);			// 计算R时的经验参数k
	int windowSize = atof(argv[3]);		// 窗口函数w(x,y)的窗口大小
	double threshold = sqrt(sqrt(0.28/255));			// 阈值

	cv::Mat Img_input = cv::imread(argv[1]);		// 读入图片

	// 定义一个Img_expand函数。三个通道BGR，初值均为0
	cv::Mat Img_expand(Img_input.size().height + 2, Img_input.size().width + 2, CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));	
	cv::Size R_Size(Img_input.size().width - windowSize / 2 * 2, Img_input.size().height - windowSize / 2 * 2);
	cv::Mat R(R_Size, CV_32F);			// 定义R值矩阵

	expand_Img_input(Img_input, Img_expand);		// 为了使得Img_input矩阵上每一个点都能做卷积求梯度，需要将输入矩阵增大2行2列
	Img_expand.convertTo(Img_expand, CV_32F);		// Imread读进来是unsigned char类型，需要转换到float便于计算，减少精度损失
	//设置R值矩阵大小，小于gre梯度函数，因为对于R中(i,j)点的梯度，根据Sobel算子要去遍历(i-window/2,j-window/2)~(i+window/2,j+window/2)位置的I

// 高斯滤波
	//GaussianBlur(Img_expand, Img_expand, cv::Size(5, 5), 1, 1);
	cv::Mat gas(5, 5, CV_32F);	// 定义高斯矩阵
	gass(gas, 1, 5);
	for (int i = 0; i < Img_expand.size().height-4; i++)
	{
		for (int j = 0; j < Img_expand.size().width-4; j++)
		{
			Img_expand.at<float>(i, j) = convolution(Img_expand, gas, i, j);	//Img_gradientX[i, j]是Img_expand[i+1, j+1]在x方向的梯度。i与j记录当前位置
		}
	}

// 计算R值							
	calculate_R(Img_expand, R, R_Size, k, windowSize, threshold);	

// 输出彩色R图
	cv::Mat R1(R_Size, CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
	getColorR(R, R1);
	cv::imshow("ColorR", R1);				
	cv::imwrite("ColorR.jpg", R1 * 255);

// 非极大值抑制，得到选取后的角点
	pick_R_local_max(R, threshold);

// 将角点以红色的点的形式绘制在原图上
	draw_corner(R, Img_input, windowSize);			// 在原图上绘制角点，windowSize用来建立原图Img_input和R两个矩阵之间坐标的联系：Img_input上对应点的坐标要比R矩阵大windowSize/2
	cv::imshow("result", Img_input);				// 由于Img_input存的是三通道BGR值，因此直接输出即可
	cv::imwrite("result.jpg", Img_input);
	cv::waitKey();

	return 0;
}