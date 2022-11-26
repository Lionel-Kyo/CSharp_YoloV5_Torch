#include "YoloV5.h"
#include <iostream>

#pragma comment(linker, "/INCLUDE:?ignore_this_library_placeholder@@YAHXZ")

struct YoloResult
{
	int ClassIndex;
	float Confidence;
	int X;
	int Y;
	int Width;
	int Height;
};

extern "C"
{
	/**
	 * Check if torch cuda is available
	 * @return result
	 */
	__declspec(dllexport) bool TorchCudaIsAvailable()
	{
		return torch::cuda::is_available();
	}

	/**
	 * Check if torch cuda cudnn is available
	 * @return result
	 */
	__declspec(dllexport) bool TorchCudaCudnnIsAvailable()
	{
		return torch::cuda::is_available();
	}

	/**
	 * Check number of cuda devices
	 * @return result
	 */

	__declspec(dllexport) int TorchCudaDeviceCount()
	{
		return torch::cuda::device_count();
	}

	__declspec(dllexport) void TorchVersion(const char** str, int* length)
	{
		std::string version = std::to_string(TORCH_VERSION_MAJOR) + ".";
		version += std::to_string(TORCH_VERSION_MINOR) + ".";
		version += std::to_string(TORCH_VERSION_PATCH);
		auto temp = new char[version.length()];
		std::memcpy(temp, version.c_str(), version.length());
		*str = temp;
		*length = version.length();
	}

	__declspec(dllexport) void CStrDelete(const char* cstr)
	{
		if (cstr != nullptr)
			delete cstr;
	}

	__declspec(dllexport) YoloV5* YoloV5NewByPath(const char* torchscriptPath, bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
	{
		return new YoloV5(torchscriptPath, isCuda, isHalf, height, width, confThres, iouThres);
	}

	__declspec(dllexport) YoloV5* YoloV5NewByArray(uint8_t* torchScriptArr, int torchScriptLength, bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
	{
		std::vector<char> buffer(torchScriptArr, torchScriptArr + torchScriptLength);
		return new YoloV5(buffer, isCuda, isHalf, height, width, confThres, iouThres);
	}

	__declspec(dllexport) void YoloV5Delete(YoloV5* yolov5)
	{
		if (yolov5 != nullptr)
			delete yolov5;
	}

	/*
	* Tensor result to YoloResults
	* @param tensorResult tensor detection result
	* @return need to be deleted, vector is created by new
	*/
	std::vector<YoloResult>* TensorToYoloResults(const torch::Tensor& tensorResult)
	{
		std::vector<YoloResult>* result = new std::vector<YoloResult>();
		for (int i = 0; i < tensorResult.size(0); i++)
		{
			int clazz = tensorResult[i][5].item().toInt();
			YoloResult item;
			item.ClassIndex = tensorResult[i][5].item().toInt();
			item.Confidence = tensorResult[i][4].item().toFloat();
			int left = tensorResult[i][0].item().toInt();
			int top = tensorResult[i][1].item().toInt();
			int right = tensorResult[i][2].item().toInt();
			int bottom = tensorResult[i][3].item().toInt();

			item.X = left;
			item.Y = top;
			item.Width = right - left;
			item.Height = bottom - top;
			result->emplace_back(item);
		}
		return result;
	}

	__declspec(dllexport) std::vector<YoloResult>* YoloV5Preditct(YoloV5* yolov5, cv::Mat* mat)
	{
		if (yolov5 == nullptr || mat == nullptr)
			return nullptr;

		try
		{
			auto prediction = yolov5->prediction(*mat);
			torch::Tensor predictionResult = prediction[0];
			std::vector<YoloResult>* result = TensorToYoloResults(predictionResult);
			return result;
		}
		catch (std::exception& ex)
		{
			std::cout << "YoloV5Preditct Exception: " << ex.what() << std::endl;
		}
		return nullptr;
	}

	__declspec(dllexport) std::vector<std::vector<YoloResult>*>* YoloV5Preditcts(YoloV5* yolov5, cv::Mat** matArr, int matArrLength)
	{
		if (yolov5 == nullptr || matArr == nullptr || matArrLength <= 0)
			return nullptr;

		try
		{
			std::vector<cv::Mat> mats;
			for (int i = 0; i < matArrLength; i++)
			{
				cv::Mat* matPtr = matArr[i];
				if (matPtr == nullptr)
					return nullptr;
				mats.emplace_back(*matPtr);
			}

			auto prediction = yolov5->prediction(mats);
			auto results = new std::vector<std::vector<YoloResult>*>();

			for (int i = 0; i < mats.size(); i++)
			{
				torch::Tensor predictionResult = prediction[i];
				std::vector<YoloResult>* result = TensorToYoloResults(predictionResult);
				results->emplace_back(result);
			}
			return results;
		}
		catch (std::exception& ex)
		{
			std::cout << "YoloV5Preditcts Exception: " << ex.what() << std::endl;
		}
		return nullptr;
	}

	__declspec(dllexport) int YoloV5ResultSize(std::vector<YoloResult>* result)
	{
		return result->size();
	}

	__declspec(dllexport) YoloResult YoloV5ResultAt(std::vector<YoloResult>* result, int at)
	{
		return result->at(at);
	}

	__declspec(dllexport) void YoloV5ResultDelete(std::vector<YoloResult>* result)
	{
		if (result != nullptr)
		{
			delete result;
		}
	}

	__declspec(dllexport) int YoloV5ResultsSize(std::vector<std::vector<YoloResult>*>* results)
	{
		return results->size();
	}

	__declspec(dllexport) std::vector<YoloResult>* YoloV5ResultsAt(std::vector<std::vector<YoloResult>*>* results, int at)
	{
		return results->at(at);
	}

	__declspec(dllexport) void YoloV5ResultsDelete(std::vector<std::vector<YoloResult>*>* results)
	{
		if (results != nullptr)
		{
			delete results;
		}
	}


	// Cv2 operation

	__declspec(dllexport) uint8_t* Cv2GetMatData(cv::Mat* matPtr, int* w, int* h, int* channel, int* imgtype)
	{
		*w = matPtr->cols;
		*h = matPtr->rows;
		*channel = matPtr->channels();
		*imgtype = 0;
		switch (matPtr->type())
		{
		case CV_8UC1:
			*imgtype = 1;
			break;
		case CV_8UC3:
			*imgtype = 3;
			break;
		case CV_8UC2:
			*imgtype = 2;
			break;
		case CV_8UC4:
			*imgtype = 4;
			break;
		}
		return matPtr->data;
	}

	__declspec(dllexport) cv::Mat* Cv2MatFromBytes(unsigned char* src, int w, int h, int channel)
	{
		int format;
		switch (channel)
		{
		case 1:
			format = CV_8UC1;
			break;
		case 2:
			format = CV_8UC2;
			break;
		case 3:
			format = CV_8UC3;
			break;
		default:
			format = CV_8UC4;
			break;
		}
		return new cv::Mat(h, w, format, src);
	}

	__declspec(dllexport) int Cv2ShowMat(const char* winName, cv::Mat* mat, int delay)
	{
		cv::imshow(winName, *mat);
		return cv::waitKey(delay);
	}

	__declspec(dllexport) void Cv2DeleteMat(cv::Mat* matPtr)
	{
		if (matPtr != nullptr)
		{
			delete matPtr;
		}
	}

}