#include "Tracker.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>

void getImageFiles(const std::string &filename, std::vector<std::string> &v);

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "Usage : VisualOdometry[.exe] images_path" << std::endl;
		return 0;
	}
	VO::Tracker tracker(std::string(argv[1]) + "/camera.yaml");
	std::vector<std::string> imageFiles;
	getImageFiles(std::string(argv[1]) + "/rgb.txt", imageFiles);
	int nImages = int(imageFiles.size());
	for (int i = 0; i < nImages; i++)
	{
		std::cout << imageFiles[i] << std::endl;
		tracker.track(cv::imread(std::string(argv[1]) + "/" + imageFiles[i], cv::IMREAD_UNCHANGED));
	}
	tracker.shutdown();
	return 0;
}

void getImageFiles(const std::string &filename, std::vector<std::string> &v)
{
	std::ifstream f;
	f.open(filename.c_str(), std::ios_base::in);
	std::string s;
	double t;
	std::string name;
	while (!f.eof())
	{
		std::getline(f, s);
		if (!s.empty())
		{
			std::stringstream ss;
			ss << s;
			ss >> t;
			ss >> name;
			v.push_back(name);
		}
	}
}