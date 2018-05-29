#include "Frame.h"
#include <opencv2/imgproc.hpp>

namespace VO
{
	/* default constructor */
	Frame::Frame() {}

	Frame::Frame(const cv::Mat &image, const cv::Ptr<cv::ORB> &orb)
	{
		/* convert color image to gray for faster keypoints detect */
		cv::Mat imageGray;
		if (image.channels() == 3)
			cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
		else if(image.channels() == 4)
			cv::cvtColor(image, imageGray, cv::COLOR_BGRA2GRAY);
		else
			imageGray = image;

		/* detect keypoints and extract descriptor */
		orb->detectAndCompute(image, cv::Mat(), _keyPoints, _descriptors);

		/* default transformation from world to camera is I */
		_Tcw = cv::Mat::eye(4, 4, CV_64F);/* default transformation from world to camera is I_4x4 */
	}

	Frame::~Frame()
	{
		_keyPoints.clear();
	}

	/* get keypoints in image */
	const std::vector<cv::KeyPoint> & Frame::getKeyPoints() const
	{
		return _keyPoints;
	}

	/* get descriptors of keypoints */
	const cv::Mat & Frame::getDescriptors() const
	{
		return _descriptors;
	}

	const cv::Mat & Frame::getTcw() const
	{
		return _Tcw;
	}

	void Frame::setTcw(const cv::Mat &Tcw)
	{
		Tcw.copyTo(_Tcw);
	}
}
