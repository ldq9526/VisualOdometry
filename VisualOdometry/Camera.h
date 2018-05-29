#ifndef VO_CAMERA
#define VO_CAMERA

#include <memory>
#include <opencv2/core.hpp>

namespace VO
{
	class Camera
	{
	private:
		/* size of images captured by camera */
		cv::Size _imageSize;

		/* intrinsic matrix and its inverse */
		cv::Mat _K, _K_inv;

	public:
		/* default constructer */
		Camera();

		Camera(const std::string &filePath);

		/* get the size of image captured by camera */
		const cv::Size & getImageSize() const;

		/* get intrinsic matrix of camera */
		const cv::Mat & getIntrinsicMatrix() const;

		/* get the inverse of intrinsic */
		const cv::Mat & getIntrinsicMatrixInv() const;
	};
}

#endif
