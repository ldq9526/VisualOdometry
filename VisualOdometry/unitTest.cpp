#include "Camera.h"
#include <iostream>

int main(int argc, char **argv)
{
	VO::Camera camera(argv[1]);
	std::cout << camera.getImageSize() << std::endl;
	std::cout << camera.getIntrinsicMatrix() << std::endl;
	std::cout << camera.getIntrinsicMatrixInv() << std::endl;
	return 0;
}