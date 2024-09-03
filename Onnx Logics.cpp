#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>


cv::Mat resizeImage(const cv::Mat& image, cv::Size size) {
	int iw = image.cols;
	int ih = image.rows;
	int w = size.width;
	int h = size.height;
	//  Get the size of the image
	double scale = std::min(static_cast<double>(w) / iw, static_cast<double>(h) / ih);
	int nw = static_cast<int>(iw * scale);
	int nh = static_cast<int>(ih * scale);
	// ����ͼ���С
	cv::Mat resizedImage;
	cv::resize(image, resizedImage, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
	// ������ͼ������ɫ����
	cv::Mat newImage = cv::Mat::zeros(h, w, image.type());
	cv::rectangle(newImage, cv::Rect(0, 0, w, h), cv::Scalar(128, 128, 128), -1);
	// ���������ͼ��������ͼ������
	int startX = (w - nw) / 2;
	int startY = (h - nh) / 2;
	resizedImage.copyTo(newImage(cv::Rect(startX, startY, nw, nh)));
	return newImage;
}



void colorizeSegmentation(const cv::Mat& score, cv::Mat& segm)
{
	const int rows = score.size[2];
	const int cols = score.size[3];
	const int chns = score.size[1];

	std::vector<cv::Vec3b> colors;
	colors.push_back(cv::Vec3b(0, 0, 0));
	colors.push_back(cv::Vec3b(128, 128, 0));
	colors.push_back(cv::Vec3b(255, 0, 0));
	colors.push_back(cv::Vec3b(0, 255, 0));
	colors.push_back(cv::Vec3b(0, 0, 255));

	cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Mat maxVal(rows, cols, CV_32FC1, score.data);
	for (int ch = 1; ch < chns; ch++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float* ptrScore = score.ptr<float>(0, ch, row);
			uint8_t* ptrMaxCl = maxCl.ptr<uint8_t>(row);
			float* ptrMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				if (ptrScore[col] > ptrMaxVal[col])
				{
					ptrMaxVal[col] = ptrScore[col];
					ptrMaxCl[col] = (uchar)ch;
				}
			}
		}
	}
	segm.create(rows, cols, CV_8UC3);
	for (int row = 0; row < rows; row++)
	{
		const uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
		cv::Vec3b* ptrSegm = segm.ptr<cv::Vec3b>(row);
		for (int col = 0; col < cols; col++)
		{
			ptrSegm[col] = colors[ptrMaxCl[col]];
		}
	}
}

cv::Mat resizeImage2(const cv::Mat& image, cv::Size size) {
	int iw = image.cols;
	int ih = image.rows;
	int w = size.width;
	int h = size.height;

	// ������ͼ������ɫ����
	// ���������ͼ��������ͼ������
	int startX = (iw - iw) / 2;
	int startY = (ih - ih * h / w) / 2;
	cv::Mat newImage = cv::Mat::zeros(ih * h / w, iw, image.type());
	image(cv::Rect(startX, startY, iw, ih * h / w)).copyTo(newImage);
	// ����ͼ���С
	cv::resize(newImage, newImage, cv::Size(w, h), 0, 0, cv::INTER_AREA);
	return newImage;
}


void UNetSearch(cv::Mat& Img, cv::Mat& mask)
{
	std::vector<cv::Point> origin, cleanC;
	cv::dnn::Net UNet;
	UNet = cv::dnn::readNetFromONNX("models.onnx");

	int l = MAX(Img.cols, Img.rows);
	float scale = 1.0 / 255;
	cv::cvtColor(Img, Img, cv::COLOR_GRAY2RGB);
	cv::Mat input = resizeImage(Img, cv::Size(512, 512));

	cv::Mat blob = cv::dnn::blobFromImage(input, scale, cv::Size(), cv::Scalar(), false, false, CV_32F);
	UNet.setInput(blob);
	std::vector<cv::Mat> output;
	UNet.forward(output, UNet.getUnconnectedOutLayersNames());
	cv::Mat segm, grayImg, threImg, addImg;

	colorizeSegmentation(output[0], segm);

	mask = cv::Mat::zeros(Img.rows, Img.cols, CV_8UC3);
	mask = resizeImage2(segm, cv::Size(2448, 2048));
	//cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
}


int main() {
	using namespace cv;
	using namespace std;

	Mat src = imread("D:\\hqy\\Project4\\test1.bmp");
	//û��ͼ������
	if (src.empty()) {
		printf("....\n");
		return -1;
	}

	cv::Mat image = cv::imread("C:\\Users\\Administrator\\Desktop1.bmp");
	cv::Size targetsize(512, 512);
cv:Mat resizedImage = resizeImage(image, targetsize);
	cv::Mat mask;
	UNetSearch(resizedImage, mask);


	//displaying result 
	cv::imshow("Original Image", image);
	cv::imshow("Resized Image", resizedImage);
	cv::imshow("Segmentation mask", mask);
	cv::waitKey(0);
	return 0;

}
