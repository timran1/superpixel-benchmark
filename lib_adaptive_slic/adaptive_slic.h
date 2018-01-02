#ifndef ADAPTIVE_SLIC_H
#define	ADAPTIVE_SLIC_H

#include <opencv2/opencv.hpp>

#include "slic.h"
#include "plotter.h"
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>


class openCL {

public:
	std::vector<cl::Platform> platforms;  // OpenCL Platforms
	std::shared_ptr<cl::Context> context;
	std::shared_ptr<cl::Program::Sources> source;
	std::shared_ptr<cl::Program> program;
	std::vector<cl::Device> devices;
	cl::Buffer iteration_error_individual;
	cl::Buffer distvec;
	cl::Buffer cluster_centers;
	cl::Buffer	labels;
	cl::Buffer image;
	cl::Buffer cluster_needs_update;
	std::map<string, std::shared_ptr<cl::Kernel>> kernels;
	std::map<string, std::shared_ptr<cl::Buffer>> buffers;
	std::vector<cl::CommandQueue> queues;
	void oclCheckError(std::string str, cl_int status) {
		if (status != CL_SUCCESS)
			std::cerr << "Error: " << str << " [" << status << "]" << std::endl;
	}

};

class AdaptiveSlic {
public:
	AdaptiveSlic(bool plot);

	// Main interface to perform SLIC.
	void compute_superpixels(const cv::Mat image, AdaptiveSlicArgs& args);

	// Reset this instance.
	void reset();

	// Get Mat containing segmentation labels.
	cv::Mat get_labels();

	// Profiler
	Profiler profiler;

public:
	cv::Mat grid_mat;

private:
	cv::Mat labels;
	cv::Mat image_mat;
	cv::Rect labels_rect;

	openCL hw;

	vector<shared_ptr<SLIC>> slics;

	// Plotting variables.
	Plotter plotter;
	bool plot;

	// Internal operations functions.
	void compute_superpixels_on_tile(shared_ptr<SLIC>, cv::Mat &labels_seg,
			AdaptiveSlicArgs& args);
};

class ImageUtils {
public:
	// Get contour mask from labels MAT
	static void get_labels_contour_mask(const cv::Mat mat, cv::Mat labels,
			cv::OutputArray _mask, bool _thick_line);

	// Adjust contrast and display MAT.
	static void show_mat(cv::Mat mat, std::string label);
};
#endif	/* SLIC_OPENCV_H */

