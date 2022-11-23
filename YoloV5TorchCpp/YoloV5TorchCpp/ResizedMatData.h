#pragma once
#ifndef RESIZEDMATDATA_H
#define RESIZEDMATDATA_H

#include <opencv2/opencv.hpp>

/**
 * ResizedMatData (class after resized cv mat image)
 */
class ResizedMatData
{
public:
	/**
	 * Constructor
	 * @param resizedMat Resized Mat Image
	 * @param originalWidth width before mat resized
	 * @param originalHeight height before mat resized
	 * @param border border size in resized mat
	 */
	ResizedMatData(const cv::Mat& resizedMat, int originalWidth, int originalHeight, int border);

	/**
	 * Create ResizedMatData
	 * @param mat original image
	 * @param height target height
	 * @param width target width
	 * @return resized image data
	 */
	ResizedMatData static resize(const cv::Mat& mat, int height, int width);

	// set resized image
	void setMat(const cv::Mat& img);

	// get resized image
	cv::Mat getMat();

	// get resized image pointer
	cv::Mat* getMatPtr();

	// Check if the original image width is larger than resized image
	bool isSmallerWidth();

	// Check if the original image height is larger than resized image
	bool isSmallerHeight();

	// get resized width
	int getWidth();

	// get resized height
	int getHeight();

	// set original image width
	void setOriginalWidth(int w);

	// get original image width
	int getOriginalWidth();

	// set original image height
	void setOriginalHeight(int h);

	// get original image width
	int getOriginalHeight();

	// get size of black border after resized
	int getBorder();
private:
	// resized image height
	int height;

	// resized image width
	int width;

	// original image width
	int originalWidth;

	// original image height
	int originalHeight;

	// size of black border after resized
	int border;

	// resized image
	cv::Mat mat;
};

#endif // !RESIZEDMATDATA_H
