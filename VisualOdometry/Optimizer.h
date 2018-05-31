#ifndef VO_OPTIMIZER
#define VO_OPTIMIZER

#include "Map.h"
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>

namespace VO
{
	class Optimizer
	{
	public:
		/* optimize result of solvePnP */
		void bundleAdjustment(
			const std::vector<cv::Point3d> &points3d,
			const std::vector<cv::Point2d> &points2d,
			const cv::Mat &K, cv::Mat &R, cv::Mat &t,
			const std::vector<unsigned long> keys,
			Map &map);

		/* optimize result of triangulation */
		void bundleAdjustment(
			std::unordered_map<int, cv::Point3d> &points3d,
			const std::vector<cv::KeyPoint> &keyPoints,
			const cv::Mat &K, cv::Mat &R, cv::Mat &t);
	};
}

#endif