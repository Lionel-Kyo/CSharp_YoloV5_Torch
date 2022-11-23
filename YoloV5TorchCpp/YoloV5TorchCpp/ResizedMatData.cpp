#include "ResizedMatData.h"

ResizedMatData::ResizedMatData(const cv::Mat& resizedMat, int originalWidth, int originalHeight, int border)
{
	this->mat = resizedMat;
	this->originalWidth = originalWidth;
	this->originalHeight = originalHeight;
	this->width = resizedMat.cols;
	this->height = resizedMat.rows;
	this->border = border;
}

ResizedMatData ResizedMatData::resize(const cv::Mat& mat, int height, int width)
{
	cv::Mat resized;
	int originalWidth = mat.cols, originalHeight = mat.rows;

	int w = originalWidth;
	int h = originalHeight;

	bool isW = (float)w / (float)h > (float)width / (float)height;


	cv::resize(mat, resized, cv::Size(
		isW ? width : (int)((float)height / (float)h * w),
		isW ? (int)((float)width / (float)w * h) : height));

	w = resized.cols, h = resized.rows;

	int border = 0;
	if (isW)
	{
		border = (height - h) / 2;
		cv::copyMakeBorder(resized, resized, (height - h) / 2, height - h - (height - h) / 2, 0, 0, cv::BORDER_CONSTANT);
	}
	else
	{
		border = (width - w) / 2;
		cv::copyMakeBorder(resized, resized, 0, 0, (width - w) / 2, width - w - (width - w) / 2, cv::BORDER_CONSTANT);
	}
	return ResizedMatData(resized, originalWidth, originalHeight, border);
}

void ResizedMatData::setMat(const cv::Mat& mat)
{
	this->mat = mat;
}

cv::Mat ResizedMatData::getMat()
{
	return this->mat;
}

cv::Mat* ResizedMatData::getMatPtr()
{
	return &this->mat;
}

bool ResizedMatData::isSmallerWidth()
{
	return (float)originalWidth / (float)originalHeight > (float)width / (float)height;
}

bool ResizedMatData::isSmallerHeight()
{
	return (float)originalHeight / (float)originalWidth > (float)height / (float)width;
}

int ResizedMatData::getWidth()
{
	return width;
}

int ResizedMatData::getHeight()
{
	return height;
}

void ResizedMatData::setOriginalWidth(int w)
{
	this->originalWidth = w;
}

int ResizedMatData::getOriginalWidth()
{
	return originalWidth;
}

void ResizedMatData::setOriginalHeight(int h)
{
	this->originalHeight = h;
}

int ResizedMatData::getOriginalHeight()
{
	return originalHeight;
}

int ResizedMatData::getBorder()
{
	return border;
}
