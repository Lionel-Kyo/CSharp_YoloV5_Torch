#include "YoloV5.h"


YoloV5::YoloV5(const std::string& torchScriptPath, bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
{
	this->model = torch::jit::load(torchScriptPath);
	this->initialize(isCuda, isHalf, height, width, confThres, iouThres);
}

YoloV5::YoloV5(const std::vector<char>& buffer, bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
{
	std::strstreambuf strStreamBuf(buffer.data(), buffer.size());
	std::istream strIs(&strStreamBuf);
	this->model = torch::jit::load(strIs);
	this->initialize(isCuda, isHalf, height, width, confThres, iouThres);
}

YoloV5::YoloV5(std::istream& stream, bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
{
	this->model = torch::jit::load(stream);
	this->initialize(isCuda, isHalf, height, width, confThres, iouThres);
}

void YoloV5::initialize(bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
{
	if (isCuda)
	{
		this->model.to(torch::kCUDA);
	}
	if (isHalf)
	{
		this->model.to(torch::kHalf);
	}
	this->height = height;
	this->width = width;
	this->isCuda = isCuda;
	this->iouThres = iouThres;
	this->confThres = confThres;
	this->isHalf = isHalf;
	this->model.eval();
	unsigned seed = time(0);
	std::srand(seed);
}

std::vector<torch::Tensor> YoloV5::non_max_suppression(const torch::Tensor& prediction, float confThres, float iouThres)
{
	torch::Tensor xc = prediction.select(2, 4) > confThres;
	int maxWh = 4096;
	int maxNms = 30000;
	std::vector<torch::Tensor> output;
	for (int i = 0; i < prediction.size(0); i++)
	{
		output.push_back(torch::zeros({ 0, 6 }));
	}
	for (int i = 0; i < prediction.size(0); i++)
	{
		torch::Tensor x = prediction[i];
		x = x.index_select(0, torch::nonzero(xc[i]).select(1, 0));
		if (x.size(0) == 0) continue;

		x.slice(1, 5, x.size(1)).mul_(x.slice(1, 4, 5));
		torch::Tensor box = xywh2xyxy(x.slice(1, 0, 4));
		std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(x.slice(1, 5, x.size(1)), 1, true);
		x = torch::cat({ box, std::get<0>(max_tuple), std::get<1>(max_tuple) }, 1);
		x = x.index_select(0, torch::nonzero(std::get<0>(max_tuple) > confThres).select(1, 0));
		int n = x.size(0);
		if (n == 0)
		{
			continue;
		}
		else if (n > maxNms)
		{
			x = x.index_select(0, x.select(1, 4).argsort(0, true).slice(0, 0, maxNms));
		}
		torch::Tensor c = x.slice(1, 5, 6) * maxWh;
		torch::Tensor boxes = x.slice(1, 0, 4) + c;
		torch::Tensor scores = x.select(1, 4);
		torch::Tensor ix = nms(boxes, scores, iouThres).to(x.device());
		output[i] = x.index_select(0, ix).cpu();
	}
	return output;
}

cv::Scalar YoloV5::getRandScalar()
{
	return cv::Scalar(std::rand() % 256, std::rand() % 256, std::rand() % 256);
}

cv::Mat YoloV5::img2RGB(const cv::Mat& img)
{
	int imgC = img.channels();
	cv::Mat result;
	if (imgC == 1)
	{
		cv::cvtColor(img, result, cv::COLOR_GRAY2RGB);
	}
	else
	{
		cv::cvtColor(img, result, cv::COLOR_BGR2RGB);
	}
	return result;
}

torch::Tensor YoloV5::img2Tensor(const cv::Mat& img)
{
	torch::Tensor data = torch::from_blob(img.data, { (int)height, (int)width, 3 }, torch::kByte);
	data = data.permute({ 2, 0, 1 });
	data = data.toType(torch::kFloat);
	data = data.div(255);
	data = data.unsqueeze(0);
	return data;
}

torch::Tensor YoloV5::xywh2xyxy(const torch::Tensor& x)
{
	torch::Tensor y = x.clone();
	y.select(1, 0) = x.select(1, 0) - x.select(1, 2) / 2;
	y.select(1, 1) = x.select(1, 1) - x.select(1, 3) / 2;
	y.select(1, 2) = x.select(1, 0) + x.select(1, 2) / 2;
	y.select(1, 3) = x.select(1, 1) + x.select(1, 3) / 2;
	return y;
}

torch::Tensor YoloV5::nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float thresh)
{
	auto x1 = bboxes.select(1, 0);
	auto y1 = bboxes.select(1, 1);
	auto x2 = bboxes.select(1, 2);
	auto y2 = bboxes.select(1, 3);
	auto areas = (x2 - x1) * (y2 - y1);
	auto tuple_sorted = scores.sort(0, true);
	auto order = std::get<1>(tuple_sorted);

	std::vector<int> keep;
	while (order.numel() > 0)
	{
		if (order.numel() == 1)
		{
			auto i = order.item();
			keep.push_back(i.toInt());
			break;
		}
		else
		{
			auto i = order[0].item();
			keep.push_back(i.toInt());
		}

		auto order_mask = order.narrow(0, 1, order.size(-1) - 1);

		auto xx1 = x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);
		auto yy1 = y1.index({ order_mask }).clamp(y1[keep.back()].item().toFloat(), 1e10);
		auto xx2 = x2.index({ order_mask }).clamp(0, x2[keep.back()].item().toFloat());
		auto yy2 = y2.index({ order_mask }).clamp(0, y2[keep.back()].item().toFloat());
		auto inter = (xx2 - xx1).clamp(0, 1e10) * (yy2 - yy1).clamp(0, 1e10);

		auto iou = inter / (areas[keep.back()] + areas.index({ order.narrow(0,1,order.size(-1) - 1) }) - inter);
		auto idx = (iou <= thresh).nonzero().squeeze();
		if (idx.numel() == 0)
		{
			break;
		}
		order = order.index({ idx + 1 });
	}
	return torch::tensor(keep);
}

std::vector<torch::Tensor> YoloV5::sizeOriginal(const std::vector<torch::Tensor>& result,
	const std::vector<ResizedMatData>& imgRDs)
{
	std::vector<torch::Tensor> resultOrg;
	for (int i = 0; i < result.size(); i++)
	{

		torch::Tensor data = result[i];
		ResizedMatData imgRD = imgRDs[i];
		for (int j = 0; j < data.size(0); j++)
		{
			torch::Tensor tensor = data.select(0, j);
			// (left, top, right, bottom)
			if (imgRD.isSmallerWidth())
			{
				tensor[1] -= imgRD.getBorder();
				tensor[3] -= imgRD.getBorder();
				tensor[0] *= (float)imgRD.getOriginalWidth() / (float)imgRD.getWidth();
				tensor[2] *= (float)imgRD.getOriginalWidth() / (float)imgRD.getWidth();
				tensor[1] *= (float)imgRD.getOriginalHeight() / (float)(imgRD.getHeight() - 2 * imgRD.getBorder());
				tensor[3] *= (float)imgRD.getOriginalHeight() / (float)(imgRD.getHeight() - 2 * imgRD.getBorder());
			}
			else
			{
				tensor[0] -= imgRD.getBorder();
				tensor[2] -= imgRD.getBorder();
				tensor[1] *= (float)imgRD.getOriginalHeight() / (float)imgRD.getHeight();
				tensor[3] *= (float)imgRD.getOriginalHeight() / (float)imgRD.getHeight();
				tensor[0] *= (float)imgRD.getOriginalWidth() / (float)(imgRD.getWidth() - 2 * imgRD.getBorder());
				tensor[2] *= (float)imgRD.getOriginalWidth() / (float)(imgRD.getWidth() - 2 * imgRD.getBorder());
			}
			// eliminate the negative number causing by the prediction result on the black border
			for (int k = 0; k < 4; k++)
			{
				if (tensor[k].item().toFloat() < 0)
				{
					tensor[k] = 0;
				}
			}
		}

		resultOrg.push_back(data);
	}
	return resultOrg;
}

std::vector<torch::Tensor> YoloV5::prediction(const torch::Tensor& data)
{
	torch::Tensor result;
	if (!data.is_cuda() && this->isCuda)
	{
		result = data.cuda();
	}
	if (data.is_cuda() && !this->isCuda)
	{
		result = data.cpu();
	}
	if (this->isHalf)
	{
		result = data.to(torch::kHalf);
	}
	torch::Tensor pred = model.forward({ result }).toTuple()->elements()[0].toTensor();
	return non_max_suppression(pred, confThres, iouThres);
}

std::vector<torch::Tensor> YoloV5::prediction(const std::string& filePath)
{
	cv::Mat img = cv::imread(filePath);
	return prediction(img);
}

std::vector<torch::Tensor> YoloV5::prediction(const cv::Mat& img)
{
	ResizedMatData imgRD = resize(img);
	cv::Mat reImg = img2RGB(imgRD.getMat());
	torch::Tensor data = img2Tensor(reImg);
	std::vector<torch::Tensor> result = prediction(data);
	std::vector<ResizedMatData> imgRDs;
	imgRDs.push_back(imgRD);
	return sizeOriginal(result, imgRDs);
}

std::vector<torch::Tensor> YoloV5::prediction(const std::vector<cv::Mat>& imgs)
{
	std::vector<ResizedMatData> imageRDs;
	std::vector<torch::Tensor> datas;
	for (int i = 0; i < imgs.size(); i++)
	{
		ResizedMatData imgRD = resize(imgs[i]);
		imageRDs.push_back(imgRD);
		cv::Mat img = img2RGB(imgRD.getMat());
		datas.push_back(img2Tensor(img));
	}
	torch::Tensor data = torch::cat(datas, 0);
	std::vector<torch::Tensor> result = prediction(data);
	return sizeOriginal(result, imageRDs);
}

ResizedMatData YoloV5::resize(const cv::Mat& img)
{
	return ResizedMatData::resize(img, height, width);
}

std::vector<ResizedMatData> YoloV5::resize(const std::vector<cv::Mat>& imgs, int height, int width)
{
	std::vector<ResizedMatData> imgRDs;
	for (int i = 0; i < imgs.size(); i++)
	{
		imgRDs.push_back(ResizedMatData::resize(imgs[i], height, width));
	}
	return imgRDs;
}

std::vector<ResizedMatData> YoloV5::resize(const std::vector<cv::Mat>& imgs)
{
	return YoloV5::resize(imgs, height, width);
}

std::vector<cv::Mat> YoloV5::drawRectangle(const std::vector<cv::Mat>& imgs,
	const std::vector<torch::Tensor>& rectangles,
	std::map<int, std::string> labels, int thickness)
{
	std::map<int, cv::Scalar> colors;
	return drawRectangle(imgs, rectangles, colors, labels, thickness);
}

std::vector<cv::Mat> YoloV5::drawRectangle(const std::vector<cv::Mat>& imgs,
	const std::vector<torch::Tensor>& rectangles,
	int thickness)
{
	std::map<int, cv::Scalar> colors;
	std::map<int, std::string> labels;
	return drawRectangle(imgs, rectangles, colors, labels, thickness);
}

std::vector<cv::Mat> YoloV5::drawRectangle(const std::vector<cv::Mat>& imgs,
	const std::vector<torch::Tensor>& rectangles,
	const std::map<int, cv::Scalar>& colors, const std::map<int, std::string>& labels, int thickness)
{
	std::vector<cv::Mat> results;
	for (int i = 0; i < imgs.size(); i++)
	{
		results.push_back(drawRectangle(imgs[i], rectangles[i], colors, labels, thickness));
	}
	return results;
}

cv::Mat YoloV5::drawRectangle(const cv::Mat& img, const torch::Tensor& rectangle, int thickness)
{
	std::map<int, cv::Scalar> colors;
	std::map<int, std::string> labels;
	return drawRectangle(img, rectangle, colors, labels, thickness);
}

cv::Mat YoloV5::drawRectangle(const cv::Mat& img, const torch::Tensor& rectangle,
	const std::map<int, std::string>& labels, int thickness)
{
	std::map<int, cv::Scalar> colors;
	return drawRectangle(img, rectangle, colors, labels, thickness);
}

cv::Mat YoloV5::drawRectangle(const cv::Mat& img, const torch::Tensor& rectangle,
	const std::map<int, cv::Scalar>& colors,
	const std::map<int, std::string>& labels, int thickness)
{
	cv::Mat result = img.clone();
	std::map<int, cv::Scalar>::const_iterator it;
	std::map<int, std::string>::const_iterator labelIt;
	for (int i = 0; i < rectangle.size(0); i++)
	{
		int clazz = rectangle[i][5].item().toInt();
		it = colors.find(clazz);
		cv::Scalar color = NULL;
		if (it == colors.end())
		{
			it = mainColors.find(clazz);
			if (it == mainColors.end())
			{
				color = getRandScalar();
				mainColors.insert(std::pair<int, cv::Scalar>(clazz, color));
			}
			else
			{
				color = it->second;
			}
		}
		else
		{
			color = it->second;
		}
		cv::rectangle(result, cv::Point(rectangle[i][0].item().toInt(), rectangle[i][1].item().toInt()), cv::Point(rectangle[i][2].item().toInt(), rectangle[i][3].item().toInt()), color, thickness);
		labelIt = labels.find(clazz);

		std::ostringstream oss;

		if (labelIt != labels.end())
		{
			oss << labelIt->second << " ";
		}

		oss << rectangle[i][4].item().toFloat();
		std::string label = oss.str();

		cv::putText(result, label, cv::Point(rectangle[i][0].item().toInt(), rectangle[i][1].item().toInt()), cv::FONT_HERSHEY_PLAIN, 1, color, thickness);
	}
	return result;
}

bool YoloV5::predictionExists(const torch::Tensor& clazz)
{
	return clazz.size(0) > 0 ? true : false;
}

bool YoloV5::predictionExists(const std::vector<torch::Tensor>& classs)
{
	for (int i = 0; i < classs.size(); i++)
	{
		if (predictionExists(classs[i]))
		{
			return true;
		}
	}
	return false;
}