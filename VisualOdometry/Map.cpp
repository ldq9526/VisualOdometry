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
}