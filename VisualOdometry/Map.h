#ifndef VO_MAP
#define VO_MAP

#include "MapPoint.h"
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>

namespace VO
{
	class Map
	{
	private:
		/* _counter increase from 0, as 3D keypoints' key */
		unsigned long _counter;

		/* storage every 3D keypoint with a key */
		std::unordered_map<unsigned long, MapPoint> _mapPoints;
	public:
		Map();

		/* insert a set of 3D keypoints */
		void insertKeyPoints(const std::unordered_map<int, cv::Point3d> &points, std::unordered_map<int, unsigned long> &pointsMap);

		/* remove a 3D keypoint by key */
		bool removeKeyPoint(unsigned long key);

		~Map();
	};
}

#endif