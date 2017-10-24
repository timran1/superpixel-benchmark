
#ifndef SLIC_UTIL_H_INCLUDED
#define SLIC_UTIL_H_INCLUDED

#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <memory>

#include <boost/timer.hpp>

typedef unsigned char byte;
typedef int word;

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

	// Outputs
	int num_clusters_updated;

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

		num_clusters_updated = 0;
	}
};


class Pixel
{
public:
	char color[3];
	word coord[2];
	Pixel ();
	Pixel (char l, char a, char b, word x, word y);
	Pixel (vector<int> & in);

	char& l () {return color[0];}
	char& a () {return color[1];}
	char& b () {return color[2];}
	word& x () {return coord[0];}
	word& y () {return coord[1];}

	std::string get_str ();
	Pixel operator+ (const Pixel & rhs) const;
	Pixel operator- (const Pixel & rhs) const;
	Pixel operator* (const Pixel & rhs) const;
	word get_mag () const;
	word get_xy_distsq_from (const Pixel & rhs);

	vector<int> get_int_arr() const;
};

class ImageRasterScan
{
protected:
	int skip;
public:
	ImageRasterScan (int skip) : skip (skip) { }
	bool is_exact_index (int index) const { return (index % skip == 0); }
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
	vector<byte> iteration_error_individual;
	vector<word> distvec;
	int iter_num;

	// For accounting only
	int num_clusters_updated;

	IterationState ();
	void init (int sz, int numlabels);
	void reset ();
};


class Profiler
{
protected:
	boost::timer timer;
	vector<pair<string, double>> checkpoints;
	bool enable;
	string name;
public:
	Profiler (string name, bool enable) : name (name), enable (enable) {}
	void reset_timer ();
	void update_checkpoint (string name);
	string print_checkpoints ();
};

#endif

