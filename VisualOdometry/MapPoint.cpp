#include "MapPoint.h"
#include <opencv2/features2d.hpp>

namespace VO
{
	MapPoint::MapPoint(const cv::Point3d &point)
	{
		_worldPoint.x = point.x;
		_worldPoint.y = point.y;
		_worldPoint.z = point.z;
	}

	const cv::Point3d & MapPoint::getWorldPoint() const
	{
		return _worldPoint;
	}

	void MapPoint::setWorldPosition(const cv::Point3d &point)
	{
		_worldPoint.x = point.x;
		_worldPoint.y = point.y;
		_worldPoint.z = point.z;
	}

	cv::Point3d MapPoint::getCameraPoint(const cv::Mat &Tcw)
	{
		cv::Mat worldPoint = (cv::Mat_<double>(4, 1) << _worldPoint.x, _worldPoint.y, _worldPoint.z, 1);
		cv::Mat cameraPoint = Tcw*worldPoint;
		return cv::Point3d(cameraPoint.at<double>(0), cameraPoint.at<double>(1), cameraPoint.at<double>(2));
	}

	cv::Point2d MapPoint::getPixelPoint(const cv::Mat &Tcw, const cv::Mat &K)
	{
		cv::Mat worldPoint = (cv::Mat_<double>(3, 1) << _worldPoint.x, _worldPoint.y, _worldPoint.z);
		cv::Mat cameraPoint = Tcw.rowRange(0,3).colRange(0,3)*worldPoint + Tcw.col(3).rowRange(0,3);
		cameraPoint /= cameraPoint.at<double>(2);
		cv::Mat pixelPoint = K*cameraPoint;
		return cv::Point2d(pixelPoint.at<double>(0), pixelPoint.at<double>(1));
	}
}