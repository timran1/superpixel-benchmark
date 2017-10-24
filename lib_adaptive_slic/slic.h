
#ifndef SLIC_H_INCLUDED
#define SLIC_H_INCLUDED

#include "slic-utils.h"
#include "opencl-slic.h"

using namespace std;

class SLIC  
{
protected:
	shared_ptr<Image> img;
	AdaptiveSlicArgs& args;

public:
	State state;
	IterationState iter_state;

	SLIC (shared_ptr<Image> img, AdaptiveSlicArgs& args);

	// Update the image this SLIC is working on.
	void set_image (shared_ptr<Image> img) { this->img = img; }

	// Initialize the iteration state.
	void init_iteration_state ();

	///	High level function to find initial seeds on the image.
	void set_initial_seeds (cv::Mat grid_mat);

	// The main SLIC algorithm for generating superpixels
	void perform_superpixel_slic_iteration ();

	// Post-processing of SLIC segmentation, to avoid stray labels.
	int enforce_labels_connectivity ();

	// Return a cv::Mat generated from state.labels.
	void get_labels_mat (cv::Mat &labels);

	// sRGB to CIELAB conversion for 2-D images
	static void do_rgb_to_lab_conversion(const cv::Mat &mat, cv::Mat &out, int padding_c_left, int padding_r_up);

private:
	// Calculate distance between two points on image.
	word calc_dist (const Pixel& p1, const Pixel& p2, float invwt);

	// Pick seeds for superpixels when step size of superpixels is given.
	void define_image_pixels_association (cv::Mat grid_mat);

	// Associate a pixel to cluster, if conditions are valid.
	inline void associate_cluster_to_pixel (int vect_index, int pixel_index, int row_start, int row_length, int cluster_num);

    // Move the superpixel seeds to low gradient positions to avoid putting seeds
    // at region boundaries.
    void perturb_seeds (const vector<float>& edges);

    // Detect color edges, to help PerturbSeeds()
    void detect_lab_edges (vector<float>& edges);

	// sRGB to XYZ conversion; helper for RGB2LAB()
	static void RGB2XYZ(const int sR, const int sG, const int sB, float& X, float& Y, float& Z);

	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	static void RGB2LAB(const int sR, const int sG, const int sB, float& lval, float& aval, float& bval);

};

#endif // !defined(_SLIC_H_INCLUDED_)
