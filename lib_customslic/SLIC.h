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
	float l;
	float a;
	float b;
	float x;
	float y;

	Pixel() : l (0), a (0), b(0), x (0), y (0) {}

	Pixel (	float l, float a, float b, float x, float y)
		: l (l), a (a), b(b), x (x), y (y)
	{ 	}

	std::string get_str ()
	{
		std::stringstream ss;
		ss << "x = " << x << ", y = " << y << ", l = " << l << ", a = " << a << ", b = " << b;
		return ss.str ();
	}

	Pixel operator+ (const Pixel & rhs)
	{
		Pixel out;
		out.l = this->l + rhs.l;
		out.a = this->a + rhs.a;
		out.b = this->b + rhs.b;
		out.x = this->x + rhs.x;
		out.y = this->y + rhs.y;

		return out;
	}

	Pixel operator* (const float & rhs)
	{
		Pixel out;
		out.l = this->l * rhs;
		out.a = this->a * rhs;
		out.b = this->b * rhs;
		out.x = this->x * rhs;
		out.y = this->y * rhs;

		return out;
	}

	float
	getXYDistSqFrom (const Pixel & rhs)
	{
		float x_diff = x - rhs.x;
		float y_diff = y - rhs.y;
		return (x_diff * x_diff) + (y_diff * y_diff);
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

class ImageRasterScan
{
public:
	int skip;

	ImageRasterScan (int skip)
	{
		this->skip = skip;
	}

	bool is_exact_index (int index)
	{
		return (index % skip == 0);
	}

	int get_near_index (int index)
	{
		return index - (index % skip);
	}
};

class State
{
public:
	vector<Pixel>				cluster_centers;
	vector<int> 				labels;
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
		updateRegionSizeFromSuperpixels (sz, args.numlabels);
	}

	void reset ()
	{
		is_init = false;
		cluster_centers.clear ();
		labels.clear ();
		region_size = 0;
	}

	void updateRegionSizeFromSuperpixels(int sz, int numlabels)
	{
		if (numlabels > 0)
			region_size = (0.5f + std::sqrt(float (sz) / (float) numlabels));
	}
};

class IterationState
{
public:
	float iteration_error;
	vector<float> clustersize;
	vector<float> distvec;
	int iter_num;

	IterationState ()
	{
		reset ();
	}

	void init (int sz, int numlabels)
	{
		clustersize.assign(numlabels, 0);
		distvec.assign(sz, DBL_MAX);
		iter_num = 0;
	}

	void reset ()
	{
		clustersize.clear ();
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
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLICIteration(int itr);

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
	static cv::Mat DoRGBtoLABConversion(const cv::Mat &mat);

private:
	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetLABXYSeeds_ForGivenStepSize(const vector<float>& edgemag);

	//============================================================================
	// Move the superpixel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void PerturbSeeds(const vector<float>& edges);

	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(vector<float>& edges);


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
