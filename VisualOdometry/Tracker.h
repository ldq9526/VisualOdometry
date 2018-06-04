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
		Frame _currentFrame, _keyFrame;

		/* a map including all 3d keypoints */
		Map _map;

		/* Optimizer */
		Optimizer _optimizer;

		/* detector and descriptor-extractor of ORB keypoints */
		cv::Ptr<cv::ORB> _orb;

		/* Brute-Force matcher, matches and mask */
		cv::BFMatcher _matcher;
		std::vector<cv::DMatch> _matches;
		cv::Mat _mask, _emptyMatrix;
	private:
		/* find matches between _keyFrame and _currentFrame by BruteForce & RANSAC
		   p1 and p2 are matched points on image , E is essential matrix */
		void findMatches(std::vector<cv::Point2d> &p1, std::vector<cv::Point2d> &p2, cv::Mat &E);

		/* intialize map , return true if at least 100 map points
		   E : 3x3 essential matrix
		   K : 3x3 camera intrinsic matrix
		   p1,p2 : matched points on image
		   [R | t] : camera pose to be estimated, Tcw = T21 = Tck
		   worldPoints : initialized map points, [index_of_keyframe_points, point3d] */
		bool initializeMap(const cv::Mat &E, const cv::Mat &K,
			const std::vector<cv::Point2d> &p1, const std::vector<cv::Point2d> &p2,
			cv::Mat &R, cv::Mat &t, std::unordered_map<int, cv::Point3d> &worldPoints);

		/* estimate camera pose by solvePnP, return true if succeed in estimating camera pose */
		bool estimatePosePnP(cv::Mat &R, cv::Mat &t, std::vector<int> &untriangulated);

	public:
		/* constructor */
		Tracker(const std::string &cameraFilePath);

		/* track new captured image */
		const cv::Mat & track(const cv::Mat &image);

		/* end tracking thread */
		void shutdown();
	};
}

#endif