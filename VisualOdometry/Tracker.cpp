#include "Tracker.h"
#include <opencv2/calib3d.hpp>
#define __DEBUG__

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
	}

	void Tracker::findCoarseMatches()
	{
		_matches.clear();
		std::vector<cv::DMatch> rawMatches;
		_matcher.match(_previousFrame.getDescriptors(), _currentFrame.getDescriptors(), rawMatches);
		double minDistance = rawMatches[0].distance;
		for (int i = 1; i<int(rawMatches.size()); i++)
			if (rawMatches[i].distance < minDistance)
				minDistance = rawMatches[i].distance;
		double distanceThreshold = std::max(2 * minDistance, 30.0);
		for (std::vector<cv::DMatch>::iterator i = rawMatches.begin(); i != rawMatches.end(); i++)
		{
			if (i->distance <= distanceThreshold)
				_matches.push_back(*i);
		}
	}

	void Tracker::track(const cv::Mat &image)
	{
		if (_state == NO_IMAGE || _state == LOST)
		{
			_previousFrame = Frame(image, _orb);
			if (_previousFrame.getKeyPoints().size() < 200) return;
			_state = NOT_INITIALIZED;
#ifdef __DEBUG__
			printf("Initializing ...\n");
#endif
		}
		else if (_state == NOT_INITIALIZED)
		{
			int nInliers;
			_currentFrame = Frame(image, _orb);
			findCoarseMatches();/* find coarse matches */
			nInliers = int(_matches.size());
			if (nInliers < 100) return;
			std::vector<cv::Point2f> p1(nInliers), p2(nInliers);/* pixel coordinates of matched keypoints */
			const std::vector<cv::KeyPoint>
				&keyPoints1 = _previousFrame.getKeyPoints(),
				&keyPoints2 = _currentFrame.getKeyPoints();
			for (int i = 0; i < nInliers; i++)
			{
				p1[i] = keyPoints1[_matches[i].queryIdx].pt;
				p2[i] = keyPoints2[_matches[i].trainIdx].pt;
			}

			/* calculate EssentialMat */
			cv::Mat K = _camera.getIntrinsicMatrix();
			cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, _mask);

			/* recover camera pose and triangulate keypoints */
			cv::Mat R, t, points3d;/* T21 = [R | t] */
			if (100 > cv::recoverPose(E, p1, p2, K, R, t, 70.0, _mask, points3d)) return;
			std::unordered_map<int, cv::Point3d> worldPoints;
			for (int i = 0; i < points3d.cols; i++)
			{
				if (!_mask.at<uchar>(i)) continue;
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
				worldPoints[_matches[i].trainIdx] = cv::Point3d(x1.at<double>(0), x1.at<double>(1), x1.at<double>(2));
			}
			if (100 > worldPoints.size()) return;

			/* [R|t] and 3D keypoints will be optimized by bundle adjustment */
			/* BundleAdjustment(); */
			/* create initial map */
			_map.insertKeyPoints(worldPoints, _currentFrame.getPointsMap());
			
			/* update _currentFrame and set it as _previousFrame */
			cv::Mat Tcw = cv::Mat::eye(4, 4, CV_64F);
			R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
			t.copyTo(Tcw.col(3).rowRange(0, 3));
			_currentFrame.setTcw(Tcw);
			_previousFrame = _currentFrame;

			/* state change to TRACKING if succeed in initializing */
			_state = TRACKING;
#ifdef __DEBUG__
			printf("Map created with %d points . Tracking start !\n", int(worldPoints.size()));
#endif
		}
		else
		{
			_currentFrame = Frame(image, _orb);
			findCoarseMatches();
			int nInliers = int(_matches.size());
			std::vector<cv::Point2f> p1(nInliers), p2(nInliers);
			const std::vector<cv::KeyPoint>
				&keyPoints1 = _previousFrame.getKeyPoints(),
				&keyPoints2 = _currentFrame.getKeyPoints();
			for (int i = 0; i < nInliers; i++)
			{
				p1[i] = keyPoints1[_matches[i].queryIdx].pt;
				p2[i] = keyPoints2[_matches[i].trainIdx].pt;
			}
			cv::Mat K = _camera.getIntrinsicMatrix();
			cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, _mask);
			std::unordered_map<int, unsigned long> &previousMap = _previousFrame.getPointsMap();
			std::unordered_map<int, unsigned long> &currentMap = _currentFrame.getPointsMap();
			std::vector<cv::Point3d> points3d;
			std::vector<cv::Point2d> points2d;
			std::vector<unsigned long> keys;/* record keys for optimizing 3D keypoints */
			std::vector<int> unTriangulatedMatches;
			for (int i = 0; i < _mask.rows; i++)
			{
				if (!_mask.at<uchar>(i)) continue;
				if (previousMap.find(_matches[i].queryIdx) != previousMap.end())
				{
					unsigned long key = previousMap[_matches[i].queryIdx];
					keys.push_back(key);
					points3d.push_back(_map.getKeyPoint(key).getWorldPoint());
					points2d.push_back(keyPoints2[_matches[i].trainIdx].pt);
					currentMap[_matches[i].trainIdx] = key;
				}
				else
					unTriangulatedMatches.push_back(i);
			}
			if (4 > keys.size())
			{
				++_lostCount;
				if (_lostCount == 5)
				{
					_state = LOST;
#ifdef __DEBUG__
					printf("Tracking lost !\n");
#endif
				}
				return;
			}
			cv::Mat r, t;
			cv::solvePnP(points3d, points2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
			_lostCount = 0;
			cv::Mat R;
			cv::Rodrigues(r, R);
			cv::Mat Tcw = cv::Mat::eye(4, 4, CV_64F);
			R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
			t.copyTo(Tcw.col(3).rowRange(0, 3));
			_currentFrame.setTcw(Tcw);
			_previousFrame = _currentFrame;
		}
	}
}