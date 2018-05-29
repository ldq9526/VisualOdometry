#ifndef VO_MAPPOINT
#define VO_MAPPOINT

#include <opencv2/core.hpp>

namespace VO
{
	class MapPoint
	{
	private:
		/* 3x1 float/double world coodinate */
		cv::Point3d _worldPoint;
	public:
		MapPoint();

		/* point is world coordinate */
		MapPoint(const cv::Point3d &point);

		const cv::Point3d & getWorldPoint() const;

		void setWorldPosition(const cv::Point3d &point);

		/* transform from world to camera coordinate */
		cv::Point3d getCameraPoint(const cv::Mat &Tcw);

		/* project to pixel coordinate */
		cv::Point2d getPixelPoint(const cv::Mat &Tcw, const cv::Mat &K);
	};
}


#endif