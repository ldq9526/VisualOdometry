#include "Optimizer.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace VO
{
	void Optimizer::bundleAdjustment(
		const std::vector<cv::Point3d> &points3d,
		const std::vector<cv::Point2d> &points2d,
		const cv::Mat &K, cv::Mat &R, cv::Mat &t)
	{
		typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
		std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
		std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
		g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));/* free */
		g2o::SparseOptimizer optimizer;
		optimizer.setAlgorithm(solver);

		/* vertex : camera pose & 3D keypoints */
		g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();/* camera pose, free */
		Eigen::Matrix3d R_mat;
		R_mat <<
			R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
			R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
			R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
		int index = 0;
		pose->setId(index++);/* vertex[0] is camera pose */
		pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2))));
		optimizer.addVertex(pose);

		std::vector<g2o::VertexSBAPointXYZ *> vertices;
		for (cv::Point3d p : points3d)
		{
			g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();/* free */
			point->setId(index++);
			point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
			point->setMarginalized(true);
			optimizer.addVertex(point);
			vertices.push_back(point);
		}

		g2o::CameraParameters *camera = new g2o::CameraParameters(
			K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);/* free */
		camera->setId(0);
		optimizer.addParameter(camera);

		/* edges */
		std::vector<g2o::EdgeProjectXYZ2UV*> edges;
		index = 1;
		for (cv::Point2d p : points2d)
		{
			g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
			edge->setId(index);
			edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
			edge->setVertex(1, pose);
			edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
			edge->setParameterId(0, 0);
			edge->setInformation(Eigen::Matrix2d::Identity());
			optimizer.addEdge(edge);
			edges.push_back(edge);
			index++;
		}

		optimizer.setVerbose(false);
		optimizer.initializeOptimization();
		optimizer.optimize(100);

		Eigen::Matrix<double, 4, 4> Tcw = Eigen::Isometry3d(pose->estimate()).matrix();
		R.at<double>(0, 0) = Tcw(0, 0); R.at<double>(0, 1) = Tcw(0, 1); R.at<double>(0, 2) = Tcw(0, 2);
		R.at<double>(1, 0) = Tcw(1, 0); R.at<double>(1, 1) = Tcw(1, 1); R.at<double>(1, 2) = Tcw(1, 2);
		R.at<double>(2, 0) = Tcw(2, 0); R.at<double>(2, 1) = Tcw(2, 1); R.at<double>(2, 2) = Tcw(2, 2);
		t.at<double>(0, 0) = Tcw(0, 3); t.at<double>(1, 0) = Tcw(1, 3); t.at<double>(2, 0) = Tcw(2, 3);

		/*for (int i = 0; i<int(vertices.size()); i++)
		{
			Eigen::Vector3d p = vertices[i]->estimate();
			map.updatePoint(keys[i], cv::Point3d(p(0), p(1), p(2)));
		}*/
	}

	void Optimizer::bundleAdjustment(
		std::unordered_map<int, cv::Point3d> &points3d,
		const std::vector<cv::DMatch> &matches,
		const std::vector<cv::KeyPoint> &keyPoints,
		const cv::Mat &K, cv::Mat &R, cv::Mat &t)
	{
		typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
		std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
		std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
		g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));/* free */
		g2o::SparseOptimizer optimizer;
		optimizer.setAlgorithm(solver);

		/* vertex : camera pose & 3D keypoints */
		g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();/* camera pose, free */
		Eigen::Matrix3d R_mat;
		R_mat <<
			R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
			R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
			R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
		int index = 0;
		pose->setId(index++);/* vertex[] is camera pose */
		pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2))));
		optimizer.addVertex(pose);

		std::vector<g2o::VertexSBAPointXYZ *> vertices;
		std::vector<int> keys;
		for (auto i = points3d.begin(); i != points3d.end(); i++)
		{
			g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();/* free */
			point->setId(index++);
			point->setEstimate(Eigen::Vector3d(i->second.x, i->second.y, i->second.z));
			point->setMarginalized(true);
			optimizer.addVertex(point);
			vertices.push_back(point);
			keys.push_back(i->first);
		}

		g2o::CameraParameters *camera = new g2o::CameraParameters(
			K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);/* free */
		camera->setId(0);
		optimizer.addParameter(camera);

		/* edges */
		index = 1;
		std::vector<g2o::EdgeProjectXYZ2UV*> edges;
		for (auto i = points3d.begin(); i != points3d.end(); i++)
		{
			g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
			edge->setId(index);
			edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
			edge->setVertex(1, pose);
			edge->setMeasurement(Eigen::Vector2d(keyPoints[matches[i->first].trainIdx].pt.x, keyPoints[matches[i->first].trainIdx].pt.y));
			edge->setParameterId(0, 0);
			edge->setInformation(Eigen::Matrix2d::Identity());
			optimizer.addEdge(edge);
			edges.push_back(edge);
			index++;
		}

		optimizer.setVerbose(false);
		optimizer.initializeOptimization();
		optimizer.optimize(100);

		Eigen::Matrix<double, 4, 4> Tcw = Eigen::Isometry3d(pose->estimate()).matrix();
		R.at<double>(0, 0) = Tcw(0, 0); R.at<double>(0, 1) = Tcw(0, 1); R.at<double>(0, 2) = Tcw(0, 2);
		R.at<double>(1, 0) = Tcw(1, 0); R.at<double>(1, 1) = Tcw(1, 1); R.at<double>(1, 2) = Tcw(1, 2);
		R.at<double>(2, 0) = Tcw(2, 0); R.at<double>(2, 1) = Tcw(2, 1); R.at<double>(2, 2) = Tcw(2, 2);
		t.at<double>(0, 0) = Tcw(0, 3); t.at<double>(1, 0) = Tcw(1, 3); t.at<double>(2, 0) = Tcw(2, 3);

		for (int i = 0; i<int(vertices.size()); i++)
		{
			Eigen::Vector3d p = vertices[i]->estimate();
			points3d[keys[i]].x = p(0);
			points3d[keys[i]].y = p(1);
			points3d[keys[i]].z = p(2);
		}
	}
}