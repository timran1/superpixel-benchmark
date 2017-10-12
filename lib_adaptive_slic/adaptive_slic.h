
#ifndef ADAPTIVE_SLIC_H
#define	ADAPTIVE_SLIC_H

#include <opencv2/opencv.hpp>

#include "slic.h"
#include "plotter.h"

class AdaptiveSlic {
public:
	AdaptiveSlic (bool plot);

	// Main interface to perform SLIC.
    void compute_superpixels (const cv::Mat image, AdaptiveSlicArgs& args);

    // Reset this instance.
    void reset ();

    // Get Mat containing segmentation labels.
    cv::Mat get_labels ();

private:
    cv::Mat labels;
	cv::Mat image_mat;
    cv::Rect labels_rect;

    vector<shared_ptr<SLIC>> slics;

    // Plotting variables.
    Plotter plotter;
    bool plot;

    // Internal operations functions.
    void compute_superpixels_on_tile (shared_ptr<SLIC>, cv::Mat &labels_seg, AdaptiveSlicArgs& args);
};


class ImageUtils
{
public:
	// Get contour mask from labels MAT
    static void get_labels_contour_mask(const cv::Mat mat,
            cv::Mat labels, cv::OutputArray _mask, bool _thick_line);

	// Adjust contrast and display MAT.
	static void show_mat (cv::Mat mat, std::string label);
};
#endif	/* SLIC_OPENCV_H */

