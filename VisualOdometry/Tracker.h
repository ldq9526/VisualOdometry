#ifndef VO_TRACKER
#define VO_TRACKER

#include "Camera.h"
#include "Frame.h"
#include "Map.h"
#include "Optimizer.h"

namespace VO
{
	class Tracker
	{
	private:
		/* the state set of tracking thread */
		enum State {
			NO_IMAGE = 0,/* when tracking thread start */
			NOT_INITIALIZED = 1,/* before getting an initial map */
			TRACKING = 2,/* while tracking thread is running normally */
			LOST = 3/* when tracking thread is lost */
		};

		/* counter of continuous lost frames */
		int _lostCount;

		/* the current state of tracking thread */
		State _state;

		/* camera in SLAM is unique */
		Camera _camera;

		/* new captured image makes _currentFrame */
		Frame _currentFrame, _previousFrame;

		/* a map including all 3d keypoints */
		Map _map;

		/* Optimizer */
		Optimizer _optimizer;

		/* detector and descriptor-extractor of ORB keypoints */
		cv::Ptr<cv::ORB> _orb;

		/* Brute-Force matcher, matches and mask */
		cv::BFMatcher _matcher;
		std::vector<cv::DMatch> _matches;
		cv::Mat _mask;
	private:
		/* find coarse matches between _previousFrame and _currentFrame by BruteForce */
		void findCoarseMatches();

		/* triangulate to add new points to map */
		void triangulate(
			const std::vector<cv::KeyPoint> &keyPoints1,
			const std::vector<cv::KeyPoint> &keyPoints2,
			std::vector<cv::DMatch> &matches, const std::vector<int> &indicies, cv::Mat &mask,
			const cv::Mat &R1, const cv::Mat &t1,
			const cv::Mat &R2, const cv::Mat &t2,
			std::unordered_map<int, cv::Point3d> &points3d);

	public:
		/* constructor */
		Tracker(const std::string &cameraFilePath);

		/* track new captured image */
		cv::Mat track(const cv::Mat &image);

		/* end tracking thread */
		void shutdown();
	};
}

#endif