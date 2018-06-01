#include "Tracker.h"
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace VO
{
	Tracker::Tracker(const std::string &cameraFilePath)
	{
		_camera = Camera(cameraFilePath);
		_orb = cv::ORB::create();
		_currentFrame = _keyFrame = nullptr;
		_state = NO_IMAGE;
		_lostCount = 0;
		_matcher = cv::BFMatcher(cv::NORM_HAMMING);
		_map = Map();
	}

	Tracker::~Tracker()
	{
		if (_keyFrame != nullptr)
			delete _keyFrame;
		if (_currentFrame != nullptr)
			delete _currentFrame;
	}

	bool Tracker::estimatePose(std::vector<cv::DMatch> &matches, cv::Mat &mask, cv::Mat &R, cv::Mat &t)
	{
		std::vector<cv::DMatch> rawMatches;
		_matcher.match(_keyFrame->getDescriptors(), _currentFrame->getDescriptors(), rawMatches);
		double minDistance = rawMatches[0].distance;
		for (int i = 1; i<int(rawMatches.size()); i++)
			if (rawMatches[i].distance < minDistance)
				minDistance = rawMatches[i].distance;
		double distanceThreshold = std::max(2 * minDistance, 30.0);
		std::vector<cv::Point2d> p1, p2;
		for (std::vector<cv::DMatch>::iterator i = rawMatches.begin(); i != rawMatches.end(); i++)
		{
			if (i->distance <= distanceThreshold)
			{
				matches.push_back(*i);
				p1.push_back(_keyFrame->getPoint2d(i->queryIdx));
				p2.push_back(_currentFrame->getPoint2d(i->trainIdx));
			}
		}
		const cv::Mat &K = _camera.getIntrinsicMatrix();
		cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, mask);
		if (_state == NOT_INITIALIZED)/* decompose E when initializing */
		{
			if (100 > cv::recoverPose(E, p1, p2, K, R, t, 70.0, mask, cv::Mat()))
				return false;
		}
		else/* solvePnP when tracking */
		{

		}
		return true;
	}

	void Tracker::triangulate()
	{

	}

	const cv::Mat & Tracker::track(const cv::Mat &image)
	{
		_currentFrame = new Frame(image, _orb);
		if (_state == NO_IMAGE || _state == LOST)
		{
			if (_currentFrame->getKeyPoints().size() < 200)
			{
				delete _currentFrame;
				_currentFrame = nullptr;
			}
			if (_keyFrame != nullptr)
			{
				delete _keyFrame;
				_keyFrame = _currentFrame;
			}
			_state = NOT_INITIALIZED;
		}
		else if(_state == NOT_INITIALIZED)
		{

		}
		else
		{

		}
		return cv::Mat();
	}

	void Tracker::shutdown()
	{
		if (_state != TRACKING) return;
		_map.saveMap();
	}
}