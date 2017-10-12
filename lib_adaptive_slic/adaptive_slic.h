
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

	// sRGB to CIELAB conversion for 2-D images
	static void do_rgb_to_lab_conversion(const cv::Mat &mat, cv::Mat &out, int padding_c_left, int padding_r_up);

	// Adjust contrast and display MAT.
	static void show_mat (cv::Mat mat, std::string label);

private:
	// sRGB to XYZ conversion; helper for RGB2LAB()
	static void RGB2XYZ(const int sR, const int sG, const int sB, float& X, float& Y, float& Z);

	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	static void RGB2LAB(const int sR, const int sG, const int sB, float& lval, float& aval, float& bval);
};
#endif	/* SLIC_OPENCV_H */

