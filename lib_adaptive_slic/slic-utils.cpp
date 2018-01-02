
#include <cmath>
#include <iostream>
#include <assert.h>

#include "slic.h"

using namespace std;

//-----------------------------------------------------------------------
// IterationState class.
//-----------------------------------------------------------------------
IterationState::IterationState ()
{
	reset ();
}

void
IterationState::init (int sz, int numlabels)
{
	distvec.assign(sz, INT_MAX);
	iteration_error_individual.assign (numlabels, 255);
	iter_num = 0;
	num_clusters_updated = 0;
}

void
IterationState::reset ()
{
	distvec.clear ();
	iteration_error_individual.clear ();
	iteration_error = FLT_MAX;
	iter_num = 0;
	num_clusters_updated = 0;
}


//-----------------------------------------------------------------------
// Image class.
//-----------------------------------------------------------------------
Image::Image (cv::Mat& mat, int width, int height)
	: width (width), height (height)
{
	data.resize (width*height);

	int ptr = 0;
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {

			data[ptr] = Pixel (
					char(mat.at<cv::Vec3b>(i,j)[0]),
					char(mat.at<cv::Vec3b>(i,j)[1]),
					char(mat.at<cv::Vec3b>(i,j)[2]),
					j,
					i);

			ptr++;
		}
	}
}


//-----------------------------------------------------------------------
// State class.
//-----------------------------------------------------------------------
void
State::init (AdaptiveSlicArgs& args, int sz)
{
	labels.assign (sz, -1);
	update_region_size_from_sp (sz, args.numlabels);
}

void
State::reset ()
{
	cluster_centers.clear ();
	labels.clear ();

	cluster_range.clear ();
	cluster_associativity_array.clear ();
	region_size = 0;
}

void
State::update_region_size_from_sp (int sz, int numlabels)
{
	if (numlabels > 0)
		region_size = (0.5f + std::sqrt(float (sz) / (float) numlabels));
}




//-----------------------------------------------------------------------
// Pixel class.
//-----------------------------------------------------------------------
Pixel::Pixel ()
{
	color[0] = 0;
	color[1] = 0;
	color[2] = 0;
	coord[0] = 0;
	coord[1] = 0;
}

Pixel::Pixel (	char l, char a, char b, int x, int y)
{
	color[0] = l;
	color[1] = a;
	color[2] = b;
	coord[0] = x;
	coord[1] = y;
}

Pixel::Pixel (vector<int> & in)
{
	color[0] = in[0];
	color[1] = in[1];
	color[2] = in[2];
	coord[0] = in[3];
	coord[1] = in[4];
}

std::string
Pixel::get_str ()
{
	std::stringstream ss;
	ss  << "l = " << color[0]
		<< ", a = " <<  color[1]
		<< ", b = " <<  color[2]
		<< ", x = " <<  coord[0]
		<< ", y = " <<  coord[1];
	return ss.str ();
}

Pixel
Pixel::operator+ (const Pixel & rhs) const
{
	Pixel out;
	for (int i=0; i<3; i++)
		out.color[i] = this->color[i] + rhs.color[i];

	for (int i=0; i<2; i++)
		out.coord[i] = this->coord[i] + rhs.coord[i];
	return out;
}

Pixel
Pixel::operator- (const Pixel & rhs) const
{
	Pixel out;
	for (int i=0; i<3; i++)
		out.color[i] = this->color[i] - rhs.color[i];

	for (int i=0; i<2; i++)
		out.coord[i] = this->coord[i] - rhs.coord[i];

	return out;
}

Pixel
Pixel::operator* (const Pixel & rhs) const
{
	Pixel out;
	for (int i=0; i<3; i++)
		out.color[i] = this->color[i] * rhs.color[i];

	for (int i=0; i<2; i++)
		out.coord[i] = this->coord[i] * rhs.coord[i];
	return out;
}

word
Pixel::get_mag () const
{
	Pixel sq = this->operator* (*this);

	word mag = 0;
	for (int i=0; i<3; i++)
		mag += sq.color[i];

	for (int i=0; i<2; i++)
		mag += sq.coord[i];

	return mag;
}

word
Pixel::get_xy_distsq_from (const Pixel & rhs)
{
	word x_diff = coord[0] - rhs.coord[0];
	word y_diff = coord[1] - rhs.coord[1];
	word out = (x_diff * x_diff) + (y_diff * y_diff);
	return out;
}

vector<int>
Pixel::get_int_arr() const
{
	vector<int> out (5);

	out[0] = color[0];
	out[1] = color[1];
	out[2] = color[2];
	out[3] = coord[0];
	out[4] = coord[1];
	return out;
}


//-----------------------------------------------------------------------
// Profiler class.
//-----------------------------------------------------------------------
void Profiler::reset_timer ()
{
	if (enable)
	{
		checkpoints.clear ();
		timer = boost::timer ();
	}
}

void Profiler::update_checkpoint (string name)
{
	if (enable)
		checkpoints.push_back (std::make_pair<string,double> (name.c_str (), timer.elapsed()));
}

string Profiler::print_checkpoints ()
{
	ostringstream ss;
	if (enable)
	{
		ss << name << endl;
		double last_time = 0;
		for (auto& point:checkpoints)
		{
			ss << "[" << point.first << "]=" << (point.second - last_time)<< "s" << endl;
			last_time = point.second;
		}
	}
	return ss.str ();
}

