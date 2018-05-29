#include "Tracker.h"
#include <opencv2/calib3d.hpp>
#define MATH_SQUARE(x) ((x)*(x))

namespace VO
{
	Tracker::Tracker(const std::string &cameraFilePath)
	{
		_camera = Camera(cameraFilePath);
		_orb = cv::ORB::create(500, 1.2f, 8);
		_state = NO_IMAGE;
		_matcher = cv::BFMatcher(cv::NORM_HAMMING);
	}

	int Tracker::triangulate(const cv::Mat &R, const cv::Mat t,
		const std::vector<cv::KeyPoint> &keyPoints1,
		const std::vector<cv::KeyPoint> &keyPoints2,
		const std::vector<cv::DMatch> &matches,
		cv::Mat &mask, std::vector<cv::Point3d> &points3d)
	{
		int nInliers = 0;
		for (int i = 0; i<int(matches.size()); i++)
		{
			if (!mask.at<uchar>(i)) continue;
			cv::Mat x1 = (cv::Mat_<double>(3, 1) << keyPoints1[matches[i].queryIdx].pt.x, keyPoints1[matches[i].queryIdx].pt.y, 1),
				x2 = (cv::Mat_<double>(3, 1) << keyPoints2[matches[i].trainIdx].pt.x, keyPoints2[matches[i].trainIdx].pt.y, 1);
			cv::Mat K_inv = _camera.getIntrinsicMatrixInv();
			x1 = K_inv*x1;
			x2 = K_inv*x2;
			cv::Mat A1 = x2.cross(R*x1), b1 = x2.cross(t);
			double L1 = MATH_SQUARE(A1.at<double>(0)) + MATH_SQUARE(A1.at<double>(1)) + MATH_SQUARE(A1.at<double>(2));
			double d1 = -A1.dot(b1) / L1;
			if (d1 < 0)
			{
				mask.at<uchar>(i) = 0;
				continue;
			}
			cv::Mat b2 = d1*R*x1 + t;
			double L2 = MATH_SQUARE(b2.at<double>(0)) + MATH_SQUARE(b2.at<double>(1)) + MATH_SQUARE(b2.at<double>(2));
			double d2 = x2.dot(b2) / L2;
			if (d2 < 0)
			{
				mask.at<uchar>(i) = 0;
				continue;
			}
			points3d.push_back(cv::Point3d(d1*x1.at<double>(0), d1*x1.at<double>(1), d1));
			++nInliers;
		}
		return nInliers;
	}

	void Tracker::initializeMap()
	{

	}

	void Tracker::track(const cv::Mat &image)
	{
		if (_state == NO_IMAGE || _state == LOST)
		{
			_previousFrame = Frame(image, _orb);
			if (_previousFrame.getKeyPoints().size() > 200)
				_state = NOT_INITIALIZED;
		}
		else if (_state == NOT_INITIALIZED)
		{
			int nInliers;
			_currentFrame = Frame(image, _orb);
			_matcher.match(_previousFrame.getDescriptors(), _currentFrame.getDescriptors(), _matches);
			nInliers = _matches.size();
			if (nInliers < 100) return;
			std::vector<cv::Point2f> p1(nInliers), p2(nInliers);
			const std::vector<cv::KeyPoint> &KP1 = _previousFrame.getKeyPoints(), &KP2 = _currentFrame.getKeyPoints();
			for (int i = 0; i < nInliers; i++)
			{
				p1[i] = KP1[_matches[i].queryIdx].pt;
				p2[i] = KP2[_matches[i].trainIdx].pt;
			}
			cv::Mat K = _camera.getIntrinsicMatrix();
			cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, _mask);
			cv::Mat R, t;/* R,t transform points in _previousFrame to _currentFrame */
			if (100 > cv::recoverPose(E, p1, p2, K, R, t, _mask)) return;
			std::vector<cv::Point3d> points3d;
			if (100 > triangulate(R, t, KP1, KP2, _matches, _mask, points3d)) return;
		}
	}
}