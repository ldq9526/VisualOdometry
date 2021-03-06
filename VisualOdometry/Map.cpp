#include "Map.h"

namespace VO
{
	Map::Map()
	{
		_counter = 0;
		_mapId = 0;
	}

	Map::~Map()
	{
		_mapPoints.clear();
	}

	void Map::insertKeyPoints(
		const std::unordered_map<int, cv::Point3d> &points,
		const std::vector<cv::DMatch> &matches,
		std::unordered_map<int, unsigned long> &pointsMap1,
		std::unordered_map<int, unsigned long> &pointsMap2)
	{
		for (auto i = points.begin(); i != points.end(); i++)
		{
			_mapPoints[_counter] = MapPoint(i->second);
			pointsMap1[matches[i->first].queryIdx] = pointsMap2[matches[i->first].trainIdx] = _counter;
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

	void Map::removeAllPoints()
	{
		_mapPoints.clear();
	}

	void Map::saveMap()
	{
		if (!_mapPoints.size())
		{
			printf("No map points !\n");
			return;
		}
		char s[10];
		sprintf(s, "%02d.ply", _mapId++);
		FILE *fp = fopen(s, "w");
		if (NULL == fp)
		{
			printf("File to save map '%s'. There may be no free disk space.\n", s);
			return;
		}
		fprintf(fp, "ply\nformat ascii 1.0\ncomment local map\n");
		fprintf(fp, "element vertex %d\n", int(_mapPoints.size()));
		fprintf(fp, "property float64 x\nproperty float64 y\nproperty float64 z\n");
		fprintf(fp, "end_header\n");
		for (auto i = _mapPoints.begin(); i != _mapPoints.end(); i++)
		{
			const cv::Point3d &p = i->second.getWorldPoint();
			fprintf(fp, "%f %f %f\n", p.x, p.y, p.z);
		}
		fclose(fp);
	}

	void Map::updatePoint(unsigned long key, const cv::Point3d &p)
	{
		_mapPoints[key].setWorldPosition(p);
	}

	const MapPoint & Map::getKeyPoint(unsigned long key)
	{
		return _mapPoints[key];
	}
}