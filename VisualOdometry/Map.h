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

		/* id of saved map, increase when map saved */
		int _mapId;

		/* storage every 3D keypoint with a key */
		std::unordered_map<unsigned long, MapPoint> _mapPoints;
	public:
		Map();

		/* insert a set of 3D keypoints */
		void insertKeyPoints(
			const std::unordered_map<int, cv::Point3d> &points,
			const std::vector<cv::DMatch> &matches,
			std::unordered_map<int, unsigned long> &pointsMap1,
			std::unordered_map<int, unsigned long> &pointsMap2);

		/* remove a 3D keypoint by key */
		bool removeKeyPoint(unsigned long key);

		/* remove all 3D keypoints */
		void removeAllPoints();

		/* save point cloud */
		void saveMap();

		/* update 3D point */
		void updatePoint(unsigned long key, const cv::Point3d &p);

		/* get a 3D point by key */
		const MapPoint & getKeyPoint(unsigned long key);

		~Map();
	};
}

#endif