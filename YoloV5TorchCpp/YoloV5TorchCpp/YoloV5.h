#pragma once
#ifndef YOLOV5_H
#define YOLOV5_H

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <ctime>
#include <strstream>
#include "ResizedMatData.h"

/**
 * YoloV5 Class
 */
class YoloV5
{
public:
	/**
	 * Constructor
	 * @param torchScriptPath YoloV5 torchscipt path
	 * @param isCuda is using Cuda (default using)
	 * @param height YoloV5 Training images' height
	 * @param width YoloV5 Training images' width
	 * @param confThres non maximum suppression's scoreThresh
	 * @param iouThres non maximum suppression's iouThresh
	 */
	YoloV5(const std::string& torchScriptPath, bool isCuda = false, bool isHalf = false,
		int height = 640, int width = 640, float confThres = 0.25, float iouThres = 0.45);

	/**
	 * Constructor
	 * @param buffer buffer of torchscript
	 * @param isCuda is using Cuda (default using)
	 * @param height YoloV5 Training images' height
	 * @param width YoloV5 Training images' width
	 * @param confThres non maximum suppression's scoreThresh
	 * @param iouThres non maximum suppression's iouThresh
	 */
	YoloV5(const std::vector<char>& buffer, bool isCuda = false, bool isHalf = false,
		int height = 640, int width = 640, float confThres = 0.25, float iouThres = 0.45);

	/**
	 * Constructor
	 * @param stream stream of torchscript
	 * @param isCuda is using Cuda (default using)
	 * @param height YoloV5 Training images' height
	 * @param width YoloV5 Training images' width
	 * @param confThres non maximum suppression's scoreThresh
	 * @param iouThres non maximum suppression's iouThresh
	 */
	YoloV5(std::istream& stream, bool isCuda = false, bool isHalf = false,
		int height = 640, int width = 640, float confThres = 0.25, float iouThres = 0.45);

	/**
	 * prediction
	 * @param data prediction data (batch, rgb, height, width)
	 */
	std::vector<torch::Tensor> prediction(const torch::Tensor& data);

	/**
	 * prediction
	 * @param filePath prediction image path
	 */
	std::vector<torch::Tensor> prediction(const std::string& filePath);

	/**
	 * prediction
	 * @param img prediction image (opencv mat)
	 */
	std::vector<torch::Tensor> prediction(const cv::Mat& img);

	/**
	 * prediction
	 * @param imgs prediction images (opencv mat)
	 */
	std::vector<torch::Tensor> prediction(const std::vector<cv::Mat>& imgs);

	/**
	 * Resize mat image
	 * @param img original image
	 * @return resized image data
	 */
	ResizedMatData resize(const cv::Mat& img);

	/**
	 * Resize mat image
	 * @param imgs images to be resized
	 * @param height target height
	 * @param width target width
	 * @return resized images data
	 */
	static std::vector<ResizedMatData> resize(const std::vector<cv::Mat>& imgs, int height, int width);

	/**
	 * Resize mat image
	 * @param imgs images to be resized
	 * @return resized images data
	 */
	std::vector<ResizedMatData> resize(const std::vector<cv::Mat>& imgs);

	/**
	 * Draw result to the images
	 * @param imgs original images
	 * @param rectangles prediction results
	 * @param labels labels in traning set
	 * @param thickness tickness of rectangles
	 * @return drawn image
	 */
	std::vector<cv::Mat> drawRectangle(const std::vector<cv::Mat>& imgs,
		const std::vector<torch::Tensor>& rectangles,
		std::map<int, std::string> labels, int thickness = 2);

	/**
	 * Draw result to the images
	 * @param imgs original images
	 * @param rectangles prediction results
	 * @param thickness tickness of rectangles
	 * @return drawn image
	 */
	std::vector<cv::Mat> drawRectangle(const std::vector<cv::Mat>& imgs,
		const std::vector<torch::Tensor>& rectangles,
		int thickness = 2);

	/**
	 * Draw result to the images
	 * @param imgs original images
	 * @param rectangles prediction results
	 * @param colors represented colour for each label
	 * @param labels labels in traning set
	 * @return drawn image
	 */
	std::vector<cv::Mat> drawRectangle(const std::vector<cv::Mat>& imgs,
		const std::vector<torch::Tensor>& rectangles,
		const std::map<int, cv::Scalar>& colors, const std::map<int, std::string>& labels, int thickness = 2);

	/**
	 * Draw result to the image
	 * @param img original image
	 * @param rectangle prediction results
	 * @param thickness tickness of rectangles
	 * @return drawn image
	 */
	cv::Mat	drawRectangle(const cv::Mat& img, const torch::Tensor& rectangle, int thickness = 2);

	/**
	 * Draw result to the image
	 * @param img original image
	 * @param rectangle prediction results
	 * @param labels labels in traning set
	 * @param thickness tickness of rectangles
	 * @return drawn image
	 */
	cv::Mat	drawRectangle(const cv::Mat& img, const torch::Tensor& rectangle,
		const std::map<int, std::string>& labels,
		int thickness = 2);

	/**
	 * Draw result to the image
	 * @param img original image
	 * @param rectangle prediction results
	 * @param colos represented colour for each label
	 * @param labels labels in traning set
	 * @param thickness tickness of rectangles
	 * @return drawn image
	 */
	cv::Mat	drawRectangle(const cv::Mat& img, const torch::Tensor& rectangle,
		const std::map<int, cv::Scalar>& colors,
		const std::map<int, std::string>& labels, int thickness = 2);

	/**
	 * Check whether prediction exists
	 * @param clazz prediction result
	 * @return return ture if the prediction result exist a valid class index
	 */
	bool predictionExists(const torch::Tensor& clazz);

	/**
	 * Check whether prediction exists
	 * @param classs prediction result
	 * @return return ture if the prediction result exist a valid class index
	 */
	bool predictionExists(const std::vector<torch::Tensor>& classs);

private:
	// is using cuda
	bool isCuda;

	// is using half precision
	bool isHalf;

	// first data clean step in non maximum suppression
	float confThres;

	// iou in non maximum suppression
	float iouThres;

	// training model height
	float height;

	// training model width
	float width;

	// map of binginding box colour
	std::map<int, cv::Scalar> mainColors;

	// torchscript model
	torch::jit::script::Module model;

	// random get a colour
	cv::Scalar getRandScalar();

	// cv mat to rgb format
	cv::Mat img2RGB(const cv::Mat& img);

	// cv mat to Tensor format
	torch::Tensor img2Tensor(const cv::Mat& img);

	// (center_x center_y w h) to (left, top, right, bottom)
	torch::Tensor xywh2xyxy(const torch::Tensor& x);

	// non maximum suppression
	torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float thresh);

	// resize back the prediction result to the orignal size
	std::vector<torch::Tensor> sizeOriginal(const std::vector<torch::Tensor>& result,
		const std::vector<ResizedMatData>& imgRDs);

	// non maximum suppression
	std::vector<torch::Tensor> non_max_suppression(const torch::Tensor& preds,
		float confThres = 0.25, float iouThres = 0.45);

	// Initialization function
	void initialize(bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres);
};

#endif // !YOLOV5_H