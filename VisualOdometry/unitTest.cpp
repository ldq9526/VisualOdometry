#include "Tracker.h"
#include <iostream>
#include <opencv2/highgui.hpp>

int main(int argc, char **argv)
{
	VO::Tracker tracker(argv[1]);
	for (int i = 2; i < argc; i++)
	{
		tracker.track(cv::imread(argv[i]));
	}
	return 0;
}