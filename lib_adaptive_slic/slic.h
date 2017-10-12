
#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_

#include "slic-utils.h"

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
	void set_initial_seeds ();

	// The main SLIC algorithm for generating superpixels
	void perform_superpixel_slic_iteration ();

	// Post-processing of SLIC segmentation, to avoid stray labels.
	int enforce_labels_connectivity ();

	// Return a cv::Mat generated from state.labels.
	void get_labels_mat (cv::Mat &labels);

private:
	// Calculate distance between two points on image.
	float calc_dist (const Pixel& p1, const Pixel& p2, float invwt);

	// Pick seeds for superpixels when step size of superpixels is given.
	void define_image_pixels_association ();

	// Associate a pixel to cluster, if conditions are valid.
	inline void associate_cluster_to_pixel (int vect_index, int pixel_index, int row_start, int row_length, int cluster_num);

    // Move the superpixel seeds to low gradient positions to avoid putting seeds
    // at region boundaries.
    void perturb_seeds (const vector<float>& edges);

    // Detect color edges, to help PerturbSeeds()
    void detect_lab_edges (vector<float>& edges);

};

#endif // !defined(_SLIC_H_INCLUDED_)
