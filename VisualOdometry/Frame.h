#ifndef VO_FRAME
#define VO_FRAME

#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace VO
{
	class Frame
	{
	private:
		/* ORB keypoints */
		std::vector<cv::KeyPoint> _keyPoints;

		/* some keypoints' and their 3D position <keypoint_idx,world_key> */
		std::unordered_map<int, unsigned long> _pointsMap;

		/* ORB descriptor */
		cv::Mat _descriptors;

		/* transformation from world to camera */
		cv::Mat _Tcw;

	public:
		/* default constructor */
		Frame();

		/* constructor
			image : input image from camera
			orb : ORB keypoints detector and descriptors extractor */
		Frame(const cv::Mat &image, const cv::Ptr<cv::ORB> &orb);

		/* get keypoints in image */
		const std::vector<cv::KeyPoint> & getKeyPoints() const;

		/* get descriptors of keypoints */
		const cv::Mat & getDescriptors() const;

		/* get transformation from world to camera */
		const cv::Mat & getTcw() const;

		/* set pose Tcw */
		void setTcw(const cv::Mat &Tcw);

		/* get keyPoint by index */
		const cv::Point2d & getPoint2d(int index) const;

		/* get keypoints' map */
		std::unordered_map<int, unsigned long> & getPointsMap();

		~Frame();
	};
}

#endif