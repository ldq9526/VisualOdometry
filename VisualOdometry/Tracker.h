#ifndef VO_TRACKER
#define VO_TRACKER

#include "Camera.h"
#include "Frame.h"

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

		/* the current state of tracking thread */
		State _state;

		/* camera in SLAM is unique */
		Camera _camera;

		/* new captured image makes _currentFrame */
		Frame _currentFrame, _previousFrame;

		/* detector and descriptor-extractor of ORB keypoints */
		cv::Ptr<cv::ORB> _orb;

		/* Brute-Force matcher , matches and mask */
		cv::BFMatcher _matcher;
		std::vector<cv::DMatch> _matches;
		cv::Mat _mask;
	private:
		/* triangulation for 3D keypoints
			[R|t] : transformation from _previousFrame to _currentFrame
			keyPoints1,keyPoints2 : ORB keypoints from _previousFrame and _currentFrame
			matches,mask : marked matches will be triangulated , outliers will be unmarked
			points3d : output of 3D keypoints
			return number of inliers */
		int triangulate(const cv::Mat &R, const cv::Mat t,
			const std::vector<cv::KeyPoint> &keyPoints1,
			const std::vector<cv::KeyPoint> &keyPoints2,
			const std::vector<cv::DMatch> &matches,
			cv::Mat &mask, std::vector<cv::Point3d> &points3d);

		/* initialize a map */
		void initializeMap();

	public:
		/* constructor */
		Tracker(const std::string &cameraFilePath);

		/* track new captured image */
		void track(const cv::Mat &image);
	};
}

#endif