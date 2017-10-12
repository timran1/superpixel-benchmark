// SLIC.h: interface for the SLIC class.
//===========================================================================
// This code implements the superpixel method described in:
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
// "SLIC Superpixels",
// EPFL Technical Report no. 149300, June 2010.
//
// Adapted to point clouds and depth by David Stutz <david.stutz@rwth-aachen.de>
//
//===========================================================================
//	Copyright (c) 2012 Radhakrishna Achanta [EPFL]. All rights reserved.
//===========================================================================
//////////////////////////////////////////////////////////////////////

#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_

#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <memory>

using namespace std;

class CUSTOMSLIC_ARGS {
public:
	// knobs
	bool stateful;
	int iterations;
	int tile_square_side;
	bool one_sided_padding;
	vector<int> access_pattern;
	int numlabels;	// = superpixels in main.cpp
	float target_error;

	// We generally do not change these.
	float compactness;
	bool perturbseeds;
	int color;

	CUSTOMSLIC_ARGS ()
	{
		// Set some default values.
		stateful = false;
		iterations = 3;
		tile_square_side = 0;
		one_sided_padding = false;
		access_pattern.resize (iterations, 1);
		numlabels = 1000;

		compactness = 40;
		perturbseeds = false;
		color = 1;
	}
};

class Pixel
{
public:
	float data[5];
	Pixel ()
	{
		data[0] = 0;
		data[1] = 0;
		data[2] = 0;
		data[3] = 0;
		data[4] = 0;
	}
	Pixel (	float l, float a, float b, float x, float y)
	{
		data[0] = l;
		data[1] = a;
		data[2] = b;
		data[3] = x;
		data[4] = y;
	}

	float& l () {return data[0];}
	float& a () {return data[1];}
	float& b () {return data[2];}
	float& x () {return data[3];}
	float& y () {return data[4];}

	std::string get_str ()
	{
		std::stringstream ss;
		ss  << "l = " << data[0]
			<< ", a = " <<  data[1]
			<< ", b = " <<  data[2]
			<< ", x = " <<  data[3]
			<< ", y = " <<  data[4];
		return ss.str ();
	}

	Pixel operator+ (const Pixel & rhs) const
	{
		Pixel out;
		for (int i=0; i<5; i++)
			out.data[i] = this->data[i] + rhs.data[i];
		return out;
	}

	Pixel operator- (const Pixel & rhs) const
	{
		Pixel out;
		for (int i=0; i<5; i++)
			out.data[i] = this->data[i] - rhs.data[i];
		return out;
	}

	Pixel operator* (const Pixel & rhs) const
	{
		Pixel out;
		for (int i=0; i<5; i++)
			out.data[i] = this->data[i] * rhs.data[i];
		return out;
	}

	Pixel operator* (const float & rhs) const
	{
		Pixel out;
		for (int i=0; i<5; i++)
			out.data[i] = this->data[i] * rhs;
		return out;
	}

	float
	getXYDistSqFrom (const Pixel & rhs)
	{
		float x_diff = data[3] - rhs.data[3];
		float y_diff = data[4] - rhs.data[4];
		return (x_diff * x_diff) + (y_diff * y_diff);
	}
};

class ImageRasterScan
{
protected:
	int skip;
public:
	ImageRasterScan (int skip) : skip (skip) { }

	bool is_exact_index (int index) { return (index % skip == 0); }
};

class State
{
public:
	vector<Pixel>				cluster_centers;
	vector<int> 				labels;
	vector<int>					associated_clusters_index;
	static const int 			CLUSTER_DIRECTIONS = 9;
	int region_size;

	bool						is_init;
	State ()
	{
		is_init = false;
		region_size = 0;
	}

	void init (CUSTOMSLIC_ARGS& args, int sz)
	{
		is_init = true;
		labels.assign (sz, -1);
		associated_clusters_index.resize (sz*9);

		updateRegionSizeFromSuperpixels (sz, args.numlabels);
	}

	void reset ()
	{
		is_init = false;
		cluster_centers.clear ();
		labels.clear ();
		associated_clusters_index.clear ();
		region_size = 0;
	}

	void updateRegionSizeFromSuperpixels(int sz, int numlabels)
	{
		if (numlabels > 0)
			region_size = (0.5f + std::sqrt(float (sz) / (float) numlabels));
	}
};


class Image
{
public:
	vector<Pixel> data;
	int	width;
	int height;
	Image (cv::Mat& mat, int width, int height)
		: width (width), height (height)
	{
		data.resize (width*height);

		int ptr = 0;
	    for (int i = 0; i < mat.rows; ++i) {
	        for (int j = 0; j < mat.cols; ++j) {

	        	data[ptr++] = Pixel (
	        			mat.at<cv::Vec3f>(i,j)[0],
						mat.at<cv::Vec3f>(i,j)[1],
						mat.at<cv::Vec3f>(i,j)[2],
						j,
						i);
	        }
	    }
	}
};

class IterationState
{
public:
	float iteration_error;
	vector<float> distvec;
	int iter_num;

	IterationState ()
	{
		reset ();
	}

	void init (int sz, int numlabels)
	{
		distvec.assign(sz, DBL_MAX);
		iter_num = 0;
	}

	void reset ()
	{
		distvec.clear ();
		iteration_error = FLT_MAX;
		iter_num = 0;
	}
};


class SLIC  
{
protected:
	shared_ptr<Image> img;
	CUSTOMSLIC_ARGS& args;
public:
	State state;
	IterationState iter_state;

	SLIC(shared_ptr<Image> img, CUSTOMSLIC_ARGS& args);
	virtual ~SLIC();

	void setImage (shared_ptr<Image> img) { this->img = img; }
	void initIterationState ()
	{
		iter_state.init (img->width * img->height, state.cluster_centers.size ());
	}

    //============================================================================
    // Move the superpixel seeds to low gradient positions to avoid putting seeds
    // at region boundaries.
    // Pick seeds for superpixels when step size of superpixels is given.^M
    //============================================================================
    void PerturbSeeds(const vector<float>& edges);

    //============================================================================
    // Detect color edges, to help PerturbSeeds()
    // Pick seeds for superpixels when step size of superpixels is given.^M
    //============================================================================
    void DetectLabEdges(vector<float>& edges);

	//============================================================================
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLICIterationPPA();

	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	int EnforceLabelConnectivity();

	//===========================================================================
	///	High level function to find initial seeds on the image.
	//===========================================================================
	void SetInitialSeeds ();

	//============================================================================
	// Return a cv::Mat generated from state.labels.
	//============================================================================
	void GetLabelsMat(cv::Mat &labels);

	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	static void DoRGBtoLABConversion(const cv::Mat &mat, cv::Mat &out, int padding_c_left, int padding_r_up);

private:
	float calc_dist (const Pixel& p1, const Pixel& p2, float invwt);

	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void DefineImagePixelsAssociation();
	inline void associate_cluster_to_pixel(int vect_index, int pixel_index, int row_start, int row_length, int cluster_num);

	//============================================================================
	// sRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	static void RGB2XYZ(
			const int&					sR,
			const int&					sG,
			const int&					sB,
			float&						X,
			float&						Y,
			float&						Z);
	//============================================================================
	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	//============================================================================
	static void RGB2LAB(
			const int&					sR,
			const int&					sG,
			const int&					sB,
			float&						lval,
			float&						aval,
			float&						bval);

};

#endif // !defined(_SLIC_H_INCLUDED_)
