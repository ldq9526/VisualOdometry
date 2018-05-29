#include "Camera.h"

namespace VO
{
	Camera::Camera() {}

	/* copy constructor */
	Camera::Camera(const Camera &camera)
	{
		_imageSize.width = camera.getImageSize().width;
		_imageSize.height = camera.getImageSize().height;
		camera.getIntrinsicMatrix().copyTo(_K);
		camera.getIntrinsicMatrixInv.copyTo(_K_inv);
	}

	Camera::Camera(const std::string &filePath)
	{
		cv::FileStorage fs;
		fs.open(filePath, cv::FileStorage::READ);
		double fx = fs["Camera.fx"];
		double fy = fs["Camera.fy"];
		double cx = fs["Camera.cx"];
		double cy = fs["Camera.cy"];
		_K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
		_K_inv = (cv::Mat_<double>(3, 3) << 1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1);
		int width = fs["Camera.width"];
		int height = fs["Camera.height"];
		_imageSize = cv::Size(width, height);
	}

	const cv::Size & Camera::getImageSize() const
	{
		return _imageSize;
	}

	const cv::Mat & Camera::getIntrinsicMatrix() const
	{
		return _K;
	}

	const cv::Mat & Camera::getIntrinsicMatrixInv() const
	{
		return _K_inv;
	}
}