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
		Frame *_currentFrame, *_keyFrame;

		/* a map including all 3d keypoints */
		Map _map;

		/* Optimizer */
		Optimizer _optimizer;

		/* detector and descriptor-extractor of ORB keypoints */
		cv::Ptr<cv::ORB> _orb;

		/* Brute-Force matcher */
		cv::BFMatcher _matcher;
	private:
		/* return true if succeed in estimating a coarse Tcw of _currentFrame,
		   and also get matches & mask between _keyFrame and _currentFrame keypoints */
		bool estimatePose(std::vector<cv::DMatch> &matches, cv::Mat &mask, cv::Mat &R, cv::Mat &t);

		/* triangulate to add new points to map */
		void triangulate();

	public:
		/* constructor */
		Tracker(const std::string &cameraFilePath);

		~Tracker();

		/* track new captured image */
		const cv::Mat & track(const cv::Mat &image);

		/* end tracking thread */
		void shutdown();
	};
}

#endif