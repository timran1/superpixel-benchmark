#include "slic.h"
#include "adaptive_slic.h"

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "superpixel_tools.h"

// OpenCL includes
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <exception>

#include "stdio.h"
using namespace cv;

AdaptiveSlic::AdaptiveSlic(bool plot) :
		plotter(5), profiler("AdaptiveSlic", true) {
	labels_rect = Rect(0, 0, 0, 0);

	this->plot = plot;
	if (plot) {
		// Set the plot
		plotter.pre_plot_cmds =
				"set multiplot layout 3, 1 title \"Adaptive SLIC\" font \",14\" \n"
						"set tmargin 3 \n";
		plotter.post_plot_cmds = "unset multiplot \n";

		// Error plot
		plotter.plots["plot1"] =
				Plot(
						"Error - avg cluster movement (per cluster per search area pixel)");
		plotter.plots["plot1"].pre_plot_cmds = "set key left bottom \n";
		plotter.plots["plot1"].series["error"] = DataSeries(" ", "Error",
				"line", 500);
		plotter.plots["plot1"].series["target"] = DataSeries(" ", "Target",
				"line", 500);

		// Iterations plot
		plotter.plots["plot2"] = Plot("Iterations");
		plotter.plots["plot2"].pre_plot_cmds = "unset key\n";
		plotter.plots["plot2"].series["iter"] = DataSeries("[ ] [0:]",
				"Iterations", "boxes fs solid ", 500);

		// Clusters saved plot
		plotter.plots["plot3"] = Plot("Num Clusters Updated");
		plotter.plots["plot3"].pre_plot_cmds = "unset key\n";
		plotter.plots["plot3"].series["clusters"] = DataSeries("[ ] [0:]",
				"Clusters", "boxes fs solid ", 500);
	}

	cl_int status;

	try {
		//-----------------------------------------------------
		// STEP 1: Discover and initialize the platforms
		//-----------------------------------------------------
		cl::Platform::get(&hw.platforms);
		if (hw.platforms.size() == 0) {
			throw "No OpenCL platforms found";
		}

		// Create a context with the first platform
		cl_context_properties cps[] = { CL_CONTEXT_PLATFORM,
				(cl_context_properties)(hw.platforms[0])(), 0 };

		// Create a context using this platform for a GPU type device
		hw.context = std::make_shared <cl::Context> (CL_DEVICE_TYPE_ALL, cps);

		// Get device list from the context
		hw.devices = hw.context->getInfo<CL_CONTEXT_DEVICES>();
		
		// Create a command queue on the first device
		hw.queues.push_back (cl::CommandQueue(*hw.context.get(), hw.devices[0], 0));

		// Read in the program source
		std::ifstream sourceFileName("/home/arslan/Documents/rw/superpixel-benchmark/lib_adaptive_slic/slic.cl");
		
		
		std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName),
				(std::istreambuf_iterator<char>()));

		hw.source = std::make_shared <cl::Program::Sources> (1,
				std::make_pair(sourceFile.c_str(), sourceFile.length() + 1));
		
		boost::filesystem::path full_path( boost::filesystem::current_path() );
		std::cout << "Current path is : " << full_path << std::endl;

		// Create the program
		hw.program = std::make_shared <cl::Program> (*(hw.context.get()), *(hw.source.get()));

		// Build the program
		status = hw.program->build(hw.devices);
		hw.oclCheckError("Build", status);

		std::cout << "Build Options:\t" << hw.program->getBuildInfo
				< CL_PROGRAM_BUILD_OPTIONS > (hw.devices[0]) << std::endl;

		std::cout << "Build Log: \n\n" << hw.program->getBuildInfo
				< CL_PROGRAM_BUILD_LOG > (hw.devices[0]) << std::endl << std::endl;
		
		

		// Create the kernel
		hw.kernels["rgb2lab"] = std::make_shared <cl::Kernel> (*(hw.program.get()), "rgb2lab");
		hw.kernels["update_check"] = std::make_shared <cl::Kernel> (*(hw.program.get()), "update_check");
		hw.kernels["distance_update"] = std::make_shared <cl::Kernel> (*(hw.program.get()), "distance_update");
		hw.kernels["distance_accumulate"] = std::make_shared <cl::Kernel> (*(hw.program.get()), "distance_accumulate");
		hw.kernels["center_update"] = std::make_shared <cl::Kernel> (*(hw.program.get()), "center_update");
		
		

	} catch (cl::Error error) {
		std::cerr << "ERROR: " << error.what() << "(" << error.err() << ")"
				<< std::endl;
		
		std::cout << "Build Options:\t" << hw.program->getBuildInfo
				< CL_PROGRAM_BUILD_OPTIONS > (hw.devices[0]) << std::endl;

		std::cout << "Build Log: \n\n" << hw.program->getBuildInfo
				< CL_PROGRAM_BUILD_LOG > (hw.devices[0]) << std::endl << std::endl;
	} catch (string e) {
		std: cerr << "Error:" << e << std::endl;
		while(1);
	}

}

void AdaptiveSlic::reset() {
	slics.clear();
	labels.release();
	labels_rect = Rect(0, 0, 0, 0);
	image_mat = cv::Mat();
	grid_mat.release();
}

cv::Mat AdaptiveSlic::get_labels() {
	// Extract out only the required region from labels
	return labels(labels_rect);
}

void AdaptiveSlic::compute_superpixels_on_tile(shared_ptr<SLIC> slic,
		cv::Mat &labels_seg, AdaptiveSlicArgs& args) {
	// Main operation
	slic->init_iteration_state();

	args.num_clusters_updated = 0;

	int itr = 0;
	for (; itr < args.iterations; itr++) {
		if (!args.parallel)
			slic->perform_superpixel_slic_iteration();
		else
			((hwSLIC*)slic.get ())->perform_superpixel_slic_iteration(hw);

		args.num_clusters_updated += slic->iter_state.num_clusters_updated;

		//cout << "Iter# " << itr << " Num Clust= " << slic->iter_state.num_clusters_updated << " Error= " << slic->iter_state.iteration_error << endl;
		stringstream ss;
		ss << "Iter " << itr << " done";
		profiler.update_checkpoint(ss.str());

		// post iteration hook.
		if (slic->iter_state.iteration_error < args.target_error)
			break;
	}

	// Post processing
	slic->enforce_labels_connectivity();

	profiler.update_checkpoint("enforce_labels_connectivity done");

	slic->get_labels_mat(labels_seg);

	profiler.update_checkpoint("get_labels_mat done");

	// Rename labels so they consist of continuously increasing
	// numbers.
	SuperpixelTools::relabelSuperpixels(labels_seg);

	profiler.update_checkpoint("relabelSuperpixels done");

}

void AdaptiveSlic::compute_superpixels(const cv::Mat mat_rgb,
		AdaptiveSlicArgs& args) {
	profiler.reset_timer();

	bool tiling = args.tile_square_side > 0;

	// Use clGetPlatformIDs() to retrieve the number of 
	// platforms

	if (!tiling) {
		// Create Mat to hold CIELAB format image.
		if (image_mat.rows != mat_rgb.rows)
			image_mat = cv::Mat::zeros(mat_rgb.rows, mat_rgb.cols, CV_8SC3);

		if (!args.parallel)
		{
		    // Convert image to CIE LAB color space.
		    SLIC::do_rgb_to_lab_conversion(mat_rgb, image_mat, 0, 0);
		}
		else
		{
			hwSLIC::do_rgb_to_lab_conversion(mat_rgb, image_mat, 0, 0, hw);
		}

		profiler.update_checkpoint("rgb_lab conv done");

		// Convert image to format suitable for SLIC algo.
		vector < shared_ptr < Image >> imgs;
		imgs.push_back(
				make_shared < Image
						> (image_mat, image_mat.cols, image_mat.rows));

		profiler.update_checkpoint("image created");

		// Create labels array to hold labels info
		if (labels.rows != image_mat.rows) {
			labels.create(image_mat.rows, image_mat.cols, CV_32SC1);
			labels_rect = Rect(0, 0, image_mat.cols, image_mat.rows);

			grid_mat.create(image_mat.rows, image_mat.cols, CV_8UC1);
		}

		// Make a new SLIC segmentor if this is first call to this function.
		if (slics.size() == 0) {
			if (args.parallel)
				slics.push_back(make_shared < SLIC > (imgs.back(), args));
			else
				slics.push_back(make_shared < hwSLIC > (imgs.back(), args));
			
			// set initial seeds
			slics.back()->set_initial_seeds(grid_mat);

			profiler.update_checkpoint("seeding done");
		} else
			// Update reference to new image for existing SLICs
			slics.back()->set_image(imgs.back());

		compute_superpixels_on_tile(slics.back(), labels, args);

		if (plot) {
			auto slic = slics.back();
			plotter.plots["plot1"].series["error"].add_point(
					slic->iter_state.iteration_error);
			plotter.plots["plot2"].series["iter"].add_point(
					slic->iter_state.iter_num);
		}
	} else {
		int square_side = args.tile_square_side;

		args.one_sided_padding = false;

		// Determine amount of padding required.
		int padding_c = square_side - (mat_rgb.cols % square_side);
		int padding_c_left = args.one_sided_padding ? 0 : padding_c / 2;
		int padding_r = square_side - (mat_rgb.rows % square_side);
		int padding_r_up = args.one_sided_padding ? 0 : padding_r / 2;

		// Create Mat to hold CIELAB format image.
		if (image_mat.rows != mat_rgb.rows)
			image_mat = cv::Mat::zeros(mat_rgb.rows + padding_r,
					mat_rgb.cols + padding_c, CV_8SC3);

		// Convert image to CIE LAB color space.
		SLIC::do_rgb_to_lab_conversion(mat_rgb, image_mat, padding_c_left,
				padding_r_up);

		// Create labels array to hold labels info
		if (labels.rows != image_mat.rows) {
			labels.create(image_mat.rows, image_mat.cols, CV_32SC1);
			labels_rect = Rect(padding_c_left, padding_r_up, mat_rgb.cols,
					mat_rgb.rows);

			// TODO: Add grid_mat support for tiling case.
			grid_mat.create(image_mat.rows, image_mat.cols, CV_8UC1);
		}

		vector < cv::Mat > mat_segs;
		vector < cv::Mat > labels_segs;
		vector < shared_ptr < Image >> imgs;

		// create square segments
		for (int r = 0; r <= (image_mat.rows - square_side); r += square_side) {
			for (int c = 0; c <= (image_mat.cols - square_side); c +=
					square_side) {
				// This only creates a reference to the bigger image (ROI)
				cv::Mat region = image_mat(
						Rect(c, r, square_side, square_side));
				cv::Mat label_region = labels(
						Rect(c, r, square_side, square_side));

				mat_segs.push_back(region);
				labels_segs.push_back(label_region);
				imgs.push_back(
						make_shared < Image
								> (region, region.cols, region.rows));
			}
		}

		int num_segments = imgs.size();

		// We do not need to increase number of superpixels to account for padding.
		// This is because args.region_size is same. args.region size is used by
		// SLIC::SetInitialSeeds to determine number of SPs to create and places
		// initial seeds accordingly.

		// Make new SLIC segmentors if this is first call to this function.
		if (slics.size() == 0) {
			for (int i = 0; i < num_segments; i++) {
				slics.push_back(make_shared < SLIC > (imgs[i], args));

				// This is the *expected* region size of each SP.
				slics.back()->state.update_region_size_from_sp(
						mat_rgb.rows * mat_rgb.cols, args.numlabels);

				// set initial seeds
				slics.back()->set_initial_seeds(grid_mat);
			}
		} else {
			// Update reference to new image for existing SLICs
			for (int i = 0; i < num_segments; i++)
				slics[i]->set_image(imgs[i]);
		}

		int num_sp_so_far = 0;
		for (int i = 0; i < num_segments; i++) {
			if (slics[i]->state.cluster_centers.size() > 0)
				compute_superpixels_on_tile(slics[i], labels_segs[i], args);
			else {
				// Scan region is bigger than the tile size. Cannot run algo, simply
				// return the whole tile as a single SP.
				labels_segs[i].setTo(cv::Scalar::all(i));
			}

			//ImageUtils::show_mat (mat_segs[i], "labels");
			//ImageUtils::show_mat (labels_segs[i], "labels");

			// Increment the label numbers with the number of SPs generated so far.
			// Post processing can only decrease number of SPs, not increase it.
			int num_sp_generated = slics[i]->state.cluster_centers.size();
			labels_segs[i] += Scalar(num_sp_so_far);
			num_sp_so_far += num_sp_generated;
		}

		//ImageUtils::show_mat (labels, "labels");

		if (plot) {
			float total_error = 0;
			float total_iter = 0;
			for (auto& slic : slics) {
				total_error += slic->iter_state.iteration_error;
				total_iter += slic->iter_state.iter_num;
			}
			plotter.plots["plot1"].series["error"].add_point(
					total_error / slics.size());
			plotter.plots["plot2"].series["iter"].add_point(
					total_iter / slics.size());
		}
	}

	// post image SLIC hook.
	if (plot) {
		plotter.plots["plot1"].series["target"].add_point(args.target_error);
		plotter.plots["plot3"].series["clusters"].add_point(
				args.num_clusters_updated);
		plotter.do_plot();
	}

}

//----------------------------------------------------------------------------
//ImageUtils function definitions.
//----------------------------------------------------------------------------
void ImageUtils::get_labels_contour_mask(const cv::Mat mat, cv::Mat labels,
		cv::OutputArray _mask, bool _thick_line) {
	// default width
	int line_width = 2;

	if (!_thick_line)
		line_width = 1;

	int m_width = mat.cols;
	int m_height = mat.rows;

	_mask.create(m_height, m_width, CV_8UC1);
	Mat mask = _mask.getMat();

	mask.setTo(0);

	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int sz = m_width * m_height;

	vector<bool> istaken(sz, false);

	int mainindex = 0;
	for (int j = 0; j < m_height; j++) {
		for (int k = 0; k < m_width; k++) {
			int np = 0;
			for (int i = 0; i < 8; i++) {
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height)) {
					int index = y * m_width + x;

					if (false == istaken[index]) {
						if (labels.at<int>(j, k) != labels.at<int>(y, x))
							np++;
					}
				}
			}
			if (np > line_width) {
				mask.at<char>(j, k) = (uchar) 255;
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}
}

void ImageUtils::show_mat(cv::Mat mat, std::string label) {
	double min_val, max_val;
	cv::minMaxIdx(mat, &min_val, &max_val);
	cv::Mat adjMap;
	cv::convertScaleAbs(mat, adjMap, 255 / max_val);
	cv::imshow(label.c_str(), adjMap);
	cv::waitKey(0);
}

