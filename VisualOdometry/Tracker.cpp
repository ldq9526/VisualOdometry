#include "Tracker.h"
#include <opencv2/calib3d.hpp>
#include <unordered_set>
#ifndef __DEBUG__
#define __DEBUG__
#include <iostream>
#endif

namespace VO
{
	Tracker::Tracker(const std::string &cameraFilePath)
	{
		_camera = Camera(cameraFilePath);
		_orb = cv::ORB::create(500, 1.2f, 8);
		_state = NO_IMAGE;
		_lostCount = 0;
		_matcher = cv::BFMatcher(cv::NORM_HAMMING);
		_map = Map();
		_emptyMatrix = cv::Mat();
	}

	void Tracker::findMatches(std::vector<cv::Point2d> &p1, std::vector<cv::Point2d> &p2, cv::Mat &E)
	{
		_matches.clear();
		std::vector<cv::DMatch> rawMatches;
		_matcher.match(_keyFrame.getDescriptors(), _currentFrame.getDescriptors(), rawMatches);
		double minDistance = rawMatches[0].distance;
		for (int i = 1; i<int(rawMatches.size()); i++)
			if (rawMatches[i].distance < minDistance)
				minDistance = rawMatches[i].distance;
		const std::vector<cv::KeyPoint> &keyPoints1 = _keyFrame.getKeyPoints(), &keyPoints2 = _currentFrame.getKeyPoints();
		double distanceThreshold = std::max(2 * minDistance, 30.0);
		/* select matches by threshold */
		for (std::vector<cv::DMatch>::iterator i = rawMatches.begin(); i != rawMatches.end(); i++)
		{
			if (i->distance <= distanceThreshold)
			{
				_matches.push_back(*i);
				const cv::Point2f &point1 = keyPoints1[i->queryIdx].pt, &point2 = keyPoints2[i->trainIdx].pt;
				p1.push_back(cv::Point2d(point1.x, point1.y));
				p2.push_back(cv::Point2d(point2.x, point2.y));
			}
		}
		const cv::Mat &K = _camera.getIntrinsicMatrix();
		/* select matches by RANSAC */
		E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, _mask);
	}

	bool Tracker::initializeMap(const cv::Mat &E, const cv::Mat &K,
		const std::vector<cv::Point2d> &p1, const std::vector<cv::Point2d> &p2,
		cv::Mat &R, cv::Mat &t, std::unordered_map<int, cv::Point3d> &worldPoints)
	{
		/* recover camera pose and triangulate keypoints */
		cv::Mat points3d;
		if (!cv::recoverPose(E, p1, p2, K, R, t, 70.0, _mask, points3d))
			return true;/* [R|t] = [I|0] */
		for (int i = 0; i < points3d.cols; i++)
		{
			if (!_mask.at<uchar>(i))
				continue;
			cv::Mat x1 = points3d.col(i).rowRange(0, 3) / points3d.at<double>(3, i);
			if (x1.at<double>(2) <= 0)
			{
				_mask.at<uchar>(i) = 0;
				continue;
			}
			cv::Mat x2 = R*x1 + t;
			if (x2.at<double>(2) <= 0)
			{
				_mask.at<uchar>(i) = 0;
				continue;
			}
			worldPoints[i] = cv::Point3d(x1.at<double>(0), x1.at<double>(1), x1.at<double>(2));
		}
		if (!worldPoints.size())
			return false;

		/* [R|t] and 3D keypoints are optimized by bundle adjustment */
		_optimizer.bundleAdjustment(worldPoints, _matches, _currentFrame.getKeyPoints(), K, R, t);
		/* create initial map */
		_map.insertKeyPoints(worldPoints, _matches, _keyFrame.getPointsMap(), _currentFrame.getPointsMap());
		return true;
	}

	bool Tracker::estimatePosePnP(cv::Mat &R, cv::Mat &t, std::vector<int> &untriangulated)
	{
		cv::Mat K = _camera.getIntrinsicMatrix();
		std::unordered_map<int, unsigned long> &keyMap = _keyFrame.getPointsMap();
		std::unordered_map<int, unsigned long> &currentMap = _currentFrame.getPointsMap();
		std::vector<cv::Point3d> points3d;/* matched 3d points for PnP */
		std::vector<cv::Point2d> points2d;/* matched 2d points for PnP */
		const std::vector<cv::KeyPoint> keyPoints2 = _currentFrame.getKeyPoints();
		std::unordered_set<int> outliers;
		for (int i = 0; i < _mask.rows; i++)
		{
			if (!_mask.at<uchar>(i)) continue;
			if (keyMap.find(_matches[i].queryIdx) != keyMap.end())
			{
				unsigned long key = keyMap[_matches[i].queryIdx];
				points3d.push_back(_map.getKeyPoint(key).getWorldPoint());
				const cv::Point2f &point = keyPoints2[_matches[i].trainIdx].pt;
				points2d.push_back(cv::Point2d(point.x, point.y));
				currentMap[_matches[i].trainIdx] = key;
				outliers.insert(i);
			}
			else
				untriangulated.push_back(i);
		}

		if (4 > outliers.size())/* too few points */
			return false;
		cv::Mat r, inliers;/* rotation */
		if (!cv::solvePnPRansac(points3d, points2d, K, cv::Mat(), r, t, false, 100, 0.4f, 0.99, inliers, cv::SOLVEPNP_EPNP))
			return false;
		cv::Rodrigues(r, R);

		/* optimize inliers by bundleAdjustment */
		std::vector<cv::Point3d> pts3d;
		std::vector<cv::Point2d> pts2d;
		for (int i = 0; i < inliers.rows; i++)
		{
			int index = inliers.at<int>(i);
			pts3d.push_back(points3d[index]);
			pts2d.push_back(points2d[index]);
			outliers.erase(index);
		}
		_optimizer.bundleAdjustment(pts3d, pts2d, K, R, t);

		/* delete outliers */
		for (auto i = outliers.begin(); i != outliers.end(); i++)
		{
			unsigned long key = keyMap[_matches[*i].queryIdx];
			_map.removeKeyPoint(key);
			keyMap.erase(key);
			_mask.at<uchar>(*i) = 0;
		}
		return true;
	}

	const cv::Mat & Tracker::track(const cv::Mat &image)
	{
		if (_state == NO_IMAGE || _state == LOST)
		{
			_keyFrame = Frame(image, _orb);
			if (100 > _keyFrame.getKeyPoints().size())
				return _emptyMatrix;
			_state = NOT_INITIALIZED;
#ifdef __DEBUG__
			std::cout << "Initializing ..." << std::endl;
#endif
			return _emptyMatrix;
		}
		else if (_state == NOT_INITIALIZED)
		{
			_currentFrame = Frame(image, _orb);

			/* find matches */
			cv::Mat E;
			std::vector<cv::Point2d> p1, p2;
			findMatches(p1, p2, E);
			if (20 > cv::countNonZero(_mask))
				return _emptyMatrix;

			/* estimate camera pose and initialize map */
			const cv::Mat &K = _camera.getIntrinsicMatrix();
			cv::Mat R, t;
			std::unordered_map<int, cv::Point3d> worldPoints;
			if (!initializeMap(E, K, p1, p2, R, t, worldPoints))
				return _emptyMatrix;
			/* return true but no inliers, [R|t] = [I|0] */
			if (!worldPoints.size()) return _keyFrame.getTcw();
			
			/* update _currentFrame */
			cv::Mat Tcw = cv::Mat::eye(4, 4, CV_64F);
			R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
			t.copyTo(Tcw.col(3).rowRange(0, 3));
			_currentFrame.setTcw(Tcw);

			/* change state to TRACKING if succeed in initializing */
			_state = TRACKING;
#ifdef __DEBUG__
			std::cout << "Tracking ..." << std::endl;
#endif
		}
		else
		{
			_currentFrame = Frame(image, _orb);
			findMatches(std::vector<cv::Point2d>(), std::vector<cv::Point2d>(), cv::Mat());
			const cv::Mat &K = _camera.getIntrinsicMatrix();
			cv::Mat R, t;
			std::vector<int> untriangulated;/* indices of untriangulated matches */
			if (!estimatePosePnP(R, t, untriangulated))
			{
				++_lostCount;
				if (_lostCount == 5)
				{
#ifdef __DEBUG__
					std::cout << "Lost !" << std::endl;
#endif
					_lostCount = 0;
					_state = LOST;
					_map.removeAllPoints();
				}
				return _emptyMatrix;
			}
		}
		return _currentFrame.getTcw();
	}

	void Tracker::shutdown()
	{
		_map.saveMap();
	}
}