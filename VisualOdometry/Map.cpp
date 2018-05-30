#include "Map.h"

namespace VO
{
	Map::Map()
	{
		_counter = 0;
	}

	Map::~Map()
	{
		_mapPoints.clear();
	}

	void Map::insertKeyPoints(const std::unordered_map<int, cv::Point3d> &points, std::unordered_map<int, unsigned long> &pointsMap)
	{
		for (auto i = points.begin(); i != points.end(); i++)
		{
			_mapPoints[_counter] = MapPoint(i->second);
			pointsMap[i->first] = _counter;
			++_counter;
		}
	}

	bool Map::removeKeyPoint(unsigned long key)
	{
		auto it = _mapPoints.find(key);
		if (it == _mapPoints.end())
			return false;
		_mapPoints.erase(it);
		return true;
	}

	MapPoint Map::getKeyPoint(unsigned long key)
	{
		return _mapPoints[key];
	}
}