
#ifndef SLIC_UTIL_H_INCLUDED
#define SLIC_UTIL_H_INCLUDED

#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <memory>

typedef unsigned char byte;
using namespace std;

class AdaptiveSlicArgs {
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

	AdaptiveSlicArgs ()
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
	Pixel ();
	Pixel (	float l, float a, float b, float x, float y);

	float& l () {return data[0];}
	float& a () {return data[1];}
	float& b () {return data[2];}
	float& x () {return data[3];}
	float& y () {return data[4];}

	std::string get_str ();
	Pixel operator+ (const Pixel & rhs) const;
	Pixel operator- (const Pixel & rhs) const;
	Pixel operator* (const Pixel & rhs) const;
	Pixel operator* (const float & rhs) const;
	float get_xy_distsq_from (const Pixel & rhs);
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
	vector<Pixel> cluster_centers;
	vector<int> labels;

	// These variables are constant for given SP num and image size.
	vector<tuple<int,int,int,int>> cluster_range;
	vector<vector<int>> cluster_associativity_array;
	int region_size = 0;

	void init (AdaptiveSlicArgs& args, int sz);
	void reset ();
	void update_region_size_from_sp (int sz, int numlabels);
};


class Image
{
public:
	vector<Pixel> data;
	int	width;
	int height;
	Image (cv::Mat& mat, int width, int height);
};

class IterationState
{
public:
	float iteration_error;
	vector<float> iteration_error_individual;
	vector<byte> distvec;
	int iter_num;

	IterationState ();
	void init (int sz, int numlabels);
	void reset ();
};

#endif

