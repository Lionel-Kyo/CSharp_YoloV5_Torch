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

	__declspec(dllexport) bool TorchCudaDeviceCount()
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

		std::vector<cv::Mat> mats;
		for (int i = 0; i < matArrLength; i++)
		{
			cv::Mat* matPtr = matArr[i];
			if (matPtr == nullptr)
				return nullptr;
			mats.emplace_back(*matPtr);
		}

		try
		{
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

	__declspec(dllexport) void* Cv2GetMatDataAndInfo(void* matPtr, int& w, int& h, int& channel, int& imgtype)
	{
		cv::Mat* img = (cv::Mat*)matPtr;
		w = img->cols;
		h = img->rows;
		channel = img->channels();
		switch (img->type())
		{
		case CV_8UC1:
			imgtype = 1;
			break;
		case CV_8UC3:
			imgtype = 3;
			break;
		case CV_8UC2:
			imgtype = 2;
			break;
		case CV_8UC4:
			imgtype = 4;
			break;
		}
		return img->data;
	}

	__declspec(dllexport) void Cv2GetMat(unsigned char* src, int w, int h, int channel, cv::Mat** mat)
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
		if (*mat != nullptr)
			delete* mat;
		*mat = new cv::Mat(h, w, format, src);
	}

	__declspec(dllexport) void Cv2DeleteMat(cv::Mat* matPtr)
	{
		if (matPtr != nullptr)
		{
			delete matPtr;
		}
	}

}


std::string LeadingZeroString(std::string str, int zeros)
{
	int number_of_zeros = zeros - str.length();

	if (number_of_zeros < 0)
		return str;

	return str.insert(0, number_of_zeros, '0');
}

std::string LeadingZeroString(int number, int zeros)
{
	return LeadingZeroString(std::to_string(number), zeros);
}

std::map<int, std::string> LoadLabels(std::string path)
{
	std::map<int, std::string> labels;
	std::ifstream f(path);
	std::string name = "";
	int i = 0;
	while (std::getline(f, name))
	{
		labels.insert(std::pair<int, std::string>(i, name));
		i++;
	}
	return labels;
}

void Predicts(YoloV5* yolo, std::map<int, std::string> labels, std::vector<std::string> imgPaths, std::string outputBasePath)
{
	int count = 1;
	for (auto& path : imgPaths)
	{
		auto img = cv::imread(path);
		cv::imshow("Test", img);
		cv::waitKey(1);
		auto result = yolo->prediction(img);
		img = yolo->drawRectangle(img, result[0], labels);
		cv::imshow("Test", img);
		cv::waitKey(1);
		std::string savePath = outputBasePath + LeadingZeroString(count, 5) + ".jpg";
		std::cout << "Saving Image to: " << savePath;
		std::cout << " status: " << cv::imwrite(savePath, img) << std::endl;
		count++;
	}
}

void Test()
{
	std::string basePath = "C:\\Users\\Administrator\\Documents\\C++Project\\YoloV5Test\\x64\\Debug\\";
	std::cout << torch::cuda::is_available() << std::endl;
	std::cout << torch::cuda::cudnn_is_available() << std::endl;
	std::cout << torch::cuda::device_count() << std::endl;
	std::cout << "PyTorch version: "
		<< TORCH_VERSION_MAJOR << "."
		<< TORCH_VERSION_MINOR << "."
		<< TORCH_VERSION_PATCH << std::endl;

	YoloV5* yolo = new YoloV5(basePath + "yolov5s.torchscript", torch::cuda::is_available());

	std::map<int, std::string> labels;
	labels.insert(std::pair<int, std::string>(0, "person"));
	labels.insert(std::pair<int, std::string>(1, "head"));

	std::string baseImagePath = "H:\\MachineLearning\\yolov5\\data\\images\\";
	std::vector<std::string> imgPaths;

	//for (int i = 1; i <= 449; i++)
	//{
	//	imgPaths.push_back(baseImagePath + LeadingZeroString(i, 5) + ".jpg");
	//}
	imgPaths.push_back("./test.jpg");

	Predicts(yolo, labels, imgPaths, "./out/");
	delete yolo;
}


extern "C" __declspec(dllexport) int main()
{
	//std::string basePath = "C:\\Users\\Administrator\\Documents\\C++Project\\YoloV5Test\\x64\\Debug\\";
	std::string basePath = "E:\\CppLib\\Yolo_V5_Test_Pt\\YoloV5_Test\\bin\\x64\\Debug\\";
	auto yolov5 = YoloV5NewByPath((basePath + "yolov5s.torchscript").c_str(), torch::cuda::is_available(), false, 640, 640, 0.25, 0.45);
	auto img = cv::imread("C:\\Users\\Lionel\\Pictures\\vlcsnap-2022-11-17-12h00m36s960.png");
	//auto result = YoloV5Preditct(yolov5, &img);
	auto mat1 = cv::imread("C:\\MLML\\yolov5\\data\\images\\00086.jpg");
	auto mat2 = cv::imread("C:\\MLML\\yolov5\\data\\images\\00093.jpg");
	cv::Mat* mats[2] = { &mat1, &mat2 };
	std::vector<cv::Mat> vecMats;
	vecMats.push_back(*mats[0]);
	vecMats.push_back(*mats[1]);
	//auto result = YoloV5Preditcts(yolov5, mats,2);
	auto result = yolov5->prediction(vecMats);
	auto outImgs = yolov5->drawRectangle(vecMats, result);

	int count = 0;
	for (auto& outImg : outImgs)
	{
		cv::imwrite(std::to_string(count) + ".jpg", outImg);
		count++;
	}

	return 0;
}

