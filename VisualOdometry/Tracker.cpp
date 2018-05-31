#include "Tracker.h"
#include <opencv2/calib3d.hpp>
#define MATH_SQUARE(x) ((x)*(x))
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

	void Tracker::triangulate(
		const std::vector<cv::KeyPoint> &keyPoints1,
		const std::vector<cv::KeyPoint> &keyPoints2,
		std::vector<cv::DMatch> &matches, const std::vector<int> &indicies, cv::Mat &mask,
		const cv::Mat &R1, const cv::Mat &t1,
		const cv::Mat &R2, const cv::Mat &t2,
		std::unordered_map<int, cv::Point3d> &points3d)
	{
		cv::Mat R = R2*R1.t(), t = t2 - R2*R1.t()*t1;
		cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);
		cv::Mat T2 = (cv::Mat_<double>(3, 4) <<
			R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
			R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
			R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
		R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
		t.copyTo(T2.col(3).rowRange(0, 3));
		cv::Mat K_inv = _camera.getIntrinsicMatrixInv();
		std::vector<cv::Point2d> p1, p2;
		for (int i = 0; i< int(indicies.size()); i++)
		{
			cv::Mat x1 = (cv::Mat_<double>(3, 1) <<
				keyPoints1[matches[indicies[i]].queryIdx].pt.x,
				keyPoints1[matches[indicies[i]].queryIdx].pt.y, 1);
			x1 = K_inv*x1;
			p1.push_back(cv::Point2d(x1.at<double>(0), x1.at<double>(1)));
			cv::Mat x2 = (cv::Mat_<double>(3, 1) <<
				keyPoints2[matches[indicies[i]].trainIdx].pt.x,
				keyPoints2[matches[indicies[i]].trainIdx].pt.y, 1);
			x2 = K_inv*x2;
			p2.push_back(cv::Point2d(x2.at<double>(0), x2.at<double>(1)));
		}
		cv::Mat pts3d;
		cv::triangulatePoints(T1, T2, p1, p2, pts3d);
		for (int i = 0; i < pts3d.cols; i++)
		{
			cv::Mat x = pts3d.col(i);
			x /= x.at<double>(3);
			points3d[matches[indicies[i]].trainIdx] = cv::Point3d(x.at<double>(0), x.at<double>(1), x.at<double>(2));
		}
	}

	cv::Mat Tracker::track(const cv::Mat &image)
	{
		if (_state == NO_IMAGE || _state == LOST)
		{
			_previousFrame = Frame(image, _orb);
			if (_previousFrame.getKeyPoints().size() < 200)
				return _previousFrame.getTcw();
			_state = NOT_INITIALIZED;
#ifdef __DEBUG__
			printf("Initializing ...\n");
#endif
			return _previousFrame.getTcw();
		}
		else if (_state == NOT_INITIALIZED)
		{
			int nInliers;
			_currentFrame = Frame(image, _orb);
			findCoarseMatches();/* find coarse matches */
			nInliers = int(_matches.size());
			if (nInliers < 100) return _previousFrame.getTcw();
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
			if (100 > cv::recoverPose(E, p1, p2, K, R, t, 70.0, _mask, points3d))
				return _previousFrame.getTcw();
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
			if (100 > worldPoints.size())
				return _previousFrame.getTcw();

			/* [R|t] and 3D keypoints are optimized by bundle adjustment */
			_optimizer.bundleAdjustment(worldPoints, _currentFrame.getKeyPoints(), K, R, t);
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
			return _previousFrame.getTcw();
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
			cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, _mask);/* eliminate wrong matches by RANSAC */
			std::unordered_map<int, unsigned long> &previousMap = _previousFrame.getPointsMap();
			std::unordered_map<int, unsigned long> &currentMap = _currentFrame.getPointsMap();
			std::vector<cv::Point3d> points3d;/* matched 3d points for PnP */
			std::vector<cv::Point2d> points2d;/* matched 2d points for PnP */
			std::vector<unsigned long> keys;/* keys of 3D keypoints */
			std::vector<int> unTriangulatedMatches;/* indices of match untriangulated */
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
					_lostCount = 0;
					_state = LOST;
					_map.removeAllPoints();
#ifdef __DEBUG__
					printf("Tracking lost !\n");
#endif
				}
				return _previousFrame.getTcw();
			}
			/* recover camera pose by PnP */
			cv::Mat r, t, R;
			cv::solvePnPRansac(points3d, points2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
			cv::Rodrigues(r, R);/* convert 3d rotation vector to 3x3 matrix */
			/* optimize 3D points and camera pose by local bundle adjustment */
			_optimizer.bundleAdjustment(points3d, points2d, K, R, t, keys, _map);
			if (double(keys.size()) / (keys.size()+unTriangulatedMatches.size()) >= 0.3)
			{
				cv::Mat Tcw(4,4,CV_64F);
				R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
				t.copyTo(Tcw.col(3).rowRange(0, 3));
				return Tcw;
			}
			/*triangulate some new points*/
			std::unordered_map<int, cv::Point3d> newPoints;
			cv::Mat Tcw = _previousFrame.getTcw();
			triangulate(keyPoints1, keyPoints2, _matches, unTriangulatedMatches, _mask,
				Tcw.rowRange(0, 3).colRange(0, 3), Tcw.col(3).rowRange(0, 3), R, t, newPoints);
			_map.insertKeyPoints(newPoints, _currentFrame.getPointsMap());
			R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
			t.copyTo(Tcw.col(3).rowRange(0, 3));
			_currentFrame.setTcw(Tcw);
			_previousFrame = _currentFrame;
#ifdef __DEBUG__
			printf("Reference Frame changed !\n");
#endif
			return Tcw;
		}
	}

	void Tracker::shutdown()
	{
#ifdef __DEBUG__
		printf("Tracking end, saving map ...");
#endif
		_map.saveMap();
#ifdef __DEBUG__
		printf(" Finished !\n");
#endif
	}
}