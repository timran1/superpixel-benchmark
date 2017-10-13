
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
	distvec.assign(sz, 255);
	iter_num = 0;
}

void
IterationState::reset ()
{
	distvec.clear ();
	iteration_error = FLT_MAX;
	iter_num = 0;
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

			data[ptr++] = Pixel (
					char(mat.at<cv::Vec3b>(i,j)[0]),
					char(mat.at<cv::Vec3b>(i,j)[1]),
					char(mat.at<cv::Vec3b>(i,j)[2]),
					j,
					i);
		}
	}
}


//-----------------------------------------------------------------------
// State class.
//-----------------------------------------------------------------------
State::State ()
{
	is_init = false;
	region_size = 0;
}

void
State::init (AdaptiveSlicArgs& args, int sz)
{
	is_init = true;
	labels.assign (sz, -1);
	associated_clusters_index.assign (sz*9, -1);

	update_region_size_from_sp (sz, args.numlabels);
}

void
State::reset ()
{
	is_init = false;
	cluster_centers.clear ();
	labels.clear ();
	associated_clusters_index.clear ();
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
	data[0] = 0;
	data[1] = 0;
	data[2] = 0;
	data[3] = 0;
	data[4] = 0;
}
Pixel::Pixel (	float l, float a, float b, float x, float y)
{
	data[0] = l;
	data[1] = a;
	data[2] = b;
	data[3] = x;
	data[4] = y;
}

std::string
Pixel::get_str ()
{
	std::stringstream ss;
	ss  << "l = " << data[0]
		<< ", a = " <<  data[1]
		<< ", b = " <<  data[2]
		<< ", x = " <<  data[3]
		<< ", y = " <<  data[4];
	return ss.str ();
}

Pixel
Pixel::operator+ (const Pixel & rhs) const
{
	Pixel out;
	for (int i=0; i<5; i++)
		out.data[i] = this->data[i] + rhs.data[i];
	return out;
}

Pixel
Pixel::operator- (const Pixel & rhs) const
{
	Pixel out;
	for (int i=0; i<5; i++)
		out.data[i] = this->data[i] - rhs.data[i];
	return out;
}

Pixel
Pixel::operator* (const Pixel & rhs) const
{
	Pixel out;
	for (int i=0; i<5; i++)
		out.data[i] = this->data[i] * rhs.data[i];
	return out;
}

Pixel
Pixel::operator* (const float & rhs) const
{
	Pixel out;
	for (int i=0; i<5; i++)
		out.data[i] = this->data[i] * rhs;
	return out;
}

float
Pixel::get_xy_distsq_from (const Pixel & rhs)
{
	float x_diff = data[3] - rhs.data[3];
	float y_diff = data[4] - rhs.data[4];
	return (x_diff * x_diff) + (y_diff * y_diff);
}
