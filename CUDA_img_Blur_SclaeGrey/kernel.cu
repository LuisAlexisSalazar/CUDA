#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;


void generateImg(unsigned char* output, int h, int w, string nameFile)
{
	Mat outData(h, w, CV_8UC1, (void*)output);
	imshow(nameFile, outData);
	//imwrite(nameFile + ".jpg", outData);
}



__global__
void blurKernel(unsigned char* image, unsigned char* output, int h, int w)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int BLUR_SIZE = 10;

	if (col < w && row < h)
	{
		int pixVal = 0;
		int pixels = 0;

		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
		{
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
			{
				int curRow = row + blurRow;
				int curCol = col + blurCol;
				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
				{
					pixVal += image[curRow * w + curCol];
					pixels++;
				}
			}
		}
		output[row * w + col] = (unsigned char)(pixVal / pixels);

	}
}




__global__
void colorToGreyscaleConversion(unsigned char* img, unsigned char* output, int h, int w, int CHANNELS)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//USING THE FORMULE WE COMPUTE FOR EACH PIXEL HIS GREY VERSION
	if (col < w && row < h)
	{
		int grey_offset = row * w + col;
		int rgb_offset = grey_offset * CHANNELS;

		unsigned char r = img[rgb_offset + 0];
		unsigned char g = img[rgb_offset + 1];
		unsigned char b = img[rgb_offset + 2];

		output[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;
	}
}



void imgGrey(unsigned char* h_image, unsigned char* h_output, int h_height, int h_width, int h_channels)
{
	unsigned char* d_img;
	unsigned char* d_imgOutput;

	cudaMalloc((void**)&d_img, h_height * h_width * h_channels);
	cudaMalloc((void**)&d_imgOutput, h_height * h_width);

	cudaMemcpy(d_img, h_image, h_height * h_width * h_channels, cudaMemcpyHostToDevice);

	dim3 Grid_image((int)ceil(h_height / 16.0), (int)ceil(h_width / 16.0));
	dim3 dimBlock(16, 16);
	colorToGreyscaleConversion <<< Grid_image, dimBlock >> > (d_img, d_imgOutput, h_height, h_width, h_channels);

	cudaMemcpy(h_output, d_imgOutput, h_height * h_width, cudaMemcpyDeviceToHost);


	cudaFree(d_imgOutput);
	cudaFree(d_img);

}






void imgBlur(unsigned char* h_image, unsigned char* h_output, int h_height, int h_width, int h_channels)
{
	unsigned char* d_img;
	unsigned char* d_imgOutput;

	cudaMalloc((void**)&d_img, h_height * h_width * h_channels);
	cudaMalloc((void**)&d_imgOutput, h_height * h_width * h_channels);


	cudaMemcpy(d_img, h_image, h_height * h_width, cudaMemcpyHostToDevice);


	dim3 Grid_image((int)ceil(h_height / 16.0), (int)ceil(h_width / 16.0));
	dim3 dimBlock(16, 16);
	blurKernel <<< Grid_image, dimBlock >> > (d_img, d_imgOutput, h_height, h_width);


	cudaMemcpy(h_output, d_imgOutput, h_height * h_width * h_channels, cudaMemcpyDeviceToHost);


	cudaFree(d_imgOutput);
	cudaFree(d_img);
}



int main()
{
	Mat img = imread("lena.jpg"); //3 canales
	Mat img2 = imread("lena.jpg", 0); // 1 canal


	cout << "Height:" << img.rows << endl;
	cout << "Width:" << img.cols << endl;
	cout << "Canales 1:" << img.channels() << endl;
	cout << "Canales 2:" << img.channels() << endl;

	unsigned char* output;
	unsigned char* output2;


	output = (unsigned char*)malloc(sizeof(unsigned char*) * img.rows * img.cols * img.channels());
	imgGrey(img.data, output, img.rows, img.cols, img.channels());


	output2 = (unsigned char*)malloc(sizeof(unsigned char*) * img2.rows * img2.cols * img2.channels());
	imgBlur(img2.data, output2, img2.rows, img2.cols, img2.channels());


	imshow("original", img);
	generateImg(output, img.rows, img.cols, "grey");
	generateImg(output2, img2.rows, img2.cols, "blur");


	waitKey();
	return 0;
}