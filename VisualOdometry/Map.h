#ifndef VO_MAP
#define VO_MAP

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

		/* storage as key-value */
		std::unordered_map<unsigned long, cv::Point3d> _mapPoints;
	public:
		Map();
		~Map();
	};
}

#endif