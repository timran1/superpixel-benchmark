// SLIC.cpp: implementation of the SLIC class.
//
// Copyright (C) Radhakrishna Achanta 2012
// All rights reserved
// Email: firstname.lastname@epfl.ch
//////////////////////////////////////////////////////////////////////
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "SLIC.h"
#include "libfixp.h"

using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC(Image& img, CUSTOMSLIC_ARGS& args)
	: img (img), args (args)
{
	state.init (args, img.width * img.height);
}

SLIC::~SLIC()
{

}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
		const int&		sR,
		const int&		sG,
		const int&		sB,
		float&			X,
		float&			Y,
		float&			Z)
{
	float R = sR/255.0;
	float G = sG/255.0;
	float B = sB/255.0;

	float r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, float& lval, float& aval, float& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	float X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	float epsilon = 0.008856;	//actual CIE standard
	float kappa   = 903.3;		//actual CIE standard

	float Xr = 0.950456;	//reference white
	float Yr = 1.0;		//reference white
	float Zr = 1.088754;	//reference white

	float xr = X/Xr;
	float yr = Y/Yr;
	float zr = Z/Zr;

	float fx, fy, fz;
	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	lval = 116.0*fy-16.0;
	aval = 500.0*(fx-fy);
	bval = 200.0*(fy-fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
cv::Mat SLIC::DoRGBtoLABConversion(const cv::Mat &mat)
{
	cv::Mat out;
	out.create(mat.rows, mat.cols, CV_32FC3);

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {

            int b = mat.at<cv::Vec3b>(i,j)[0];
            int g = mat.at<cv::Vec3b>(i,j)[1];
            int r = mat.at<cv::Vec3b>(i,j)[2];

            int arr_index = j + mat.cols*i;

            float l_out, a_out, b_out;
    		RGB2LAB( r, g, b, l_out, a_out, b_out);

    		out.at<cv::Vec3f>(i,j)[0] = l_out;
    		out.at<cv::Vec3f>(i,j)[1] = a_out;
    		out.at<cv::Vec3f>(i,j)[2] = b_out;
        }
    }
    return out;
}

//==============================================================================
///	GetLabelsMat
//==============================================================================
void SLIC::GetLabelsMat(cv::Mat &labels)
{
	// Convert labels.
	labels.create(img.height, img.width, CV_32SC1);
	for (int i = 0; i < img.height; ++i) {
		for (int j = 0; j < img.width; ++j) {
			labels.at<int>(i, j) = state.labels[j + i*img.width];
		}
	}
}

//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges (vector<float>& edges)
{
	int& width = img.width;
	int& height = img.height;

	int sz = width*height;

	edges.resize(sz,0);
	for( int j = 1; j < img.height-1; j++ )
	{
		for( int k = 1; k < img.width-1; k++ )
		{
			int i = j*img.width+k;

			float dx = (img.data[i-1].l-img.data[i+1].l)*(img.data[i-1].l-img.data[i+1].l) +
					(img.data[i-1].a-img.data[i+1].a)*(img.data[i-1].a-img.data[i+1].a) +
					(img.data[i-1].b-img.data[i+1].b)*(img.data[i-1].b-img.data[i+1].b);

			float dy = (img.data[i-width].l-img.data[i+width].l)*(img.data[i-width].l-img.data[i+width].l) +
					(img.data[i-width].a-img.data[i+width].a)*(img.data[i-width].a-img.data[i+width].a) +
					(img.data[i-width].b-img.data[i+width].b)*(img.data[i-width].b-img.data[i+width].b);

			//edges[i] = fabs(dx) + fabs(dy);
			edges[i] = dx*dx + dy*dy;
		}
	}
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLIC::PerturbSeeds(const vector<float>& edges)
{
	vector<Pixel>& cluster_centers = state.cluster_centers;

	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	int numseeds = cluster_centers.size();

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = cluster_centers[n].x;//original x
		int oy = cluster_centers[n].y;//original y
		int oind = oy*img.width + ox;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < img.width && ny >= 0 && ny < img.height)
			{
				int nind = ny*img.width + nx;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if(storeind != oind)
		{
			cluster_centers[n] = Pixel (
					img.data[storeind].l,
					img.data[storeind].a,
					img.data[storeind].b,
					storeind%img.width,
					storeind/img.width);
		}
	}
}

//===========================================================================
///	SetInitialSeeds
///
/// High level function to find initial seeds on the image.
//===========================================================================
void SLIC::SetInitialSeeds ()
{
	vector<float> edgemag(0);
	if(args.perturbseeds)
		DetectLabEdges(edgemag);
	GetLABXYSeeds_ForGivenStepSize(edgemag);

	// compute region size for number of clusters actually created. args.region_size is
    // used to determine search area during SLIC iterations.
	state.updateRegionSizeFromSuperpixels (img.width *img.height, state.cluster_centers.size ());
}

//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenStepSize(const vector<float>& edgemag)
{
	vector<Pixel>& cluster_centers = state.cluster_centers;
	const int STEP = state.region_size;
	const bool perturbseeds = args.perturbseeds;

	const bool hexgrid = false;
	int numseeds(0);
	int n(0);

	//int xstrips = m_width/STEP;
	//int ystrips = m_height/STEP;
	int xstrips = (0.5+float(img.width)/float(STEP));
	int ystrips = (0.5+float(img.height)/float(STEP));

	int xerr = img.width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = img.width - STEP*xstrips;}
	int yerr = img.height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = img.height- STEP*ystrips;}

	float xerrperstrip = float(xerr)/float(xstrips);
	float yerrperstrip = float(yerr)/float(ystrips);

	int xoff = STEP/2;
	int yoff = STEP/2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	cluster_centers.resize (numseeds);

	for( int y = 0; y < ystrips; y++ )
	{
		int ye = y*yerrperstrip;
		for( int x = 0; x < xstrips; x++ )
		{
			int xe = x*xerrperstrip;
			int seedx = (x*STEP+xoff+xe);
			if(hexgrid){ seedx = x*STEP+(xoff<<(y&0x1))+xe; seedx = min(img.width-1,seedx); }//for hex grid sampling
			int seedy = (y*STEP+yoff+ye);
			int i = seedy*img.width + seedx;

			cluster_centers[n] = Pixel (
					img.data[i].l,
					img.data[i].a,
					img.data[i].b,
					seedx,
					seedy
					);
			n++;
		}
	}

	if(perturbseeds)
	{
		PerturbSeeds(edgemag);
	}
}

//#define APPROXIMATION

#ifdef APPROXIMATION
inline void assign_fixed_vector (vector<fixedp>& vect, int size, float val)
{
	vect.resize (size);
	for (int i=0; i<vect.size(); i++)
	{
		fixedp temp = fixedp (vect[0].integer, vect[0].fraction, val);
		vect[i].fraction = temp.fraction;
		vect[i].integer = temp.integer;
		vect[i].value = temp.value;
	}

}
#endif

//===========================================================================
///	PerformSuperpixelSLIC
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
//===========================================================================
void SLIC::PerformSuperpixelSLIC()
{
	const int STEP = state.region_size;

	int sz = img.width*img.height;
	const int numk = state.cluster_centers.size ();


#ifdef APPROXIMATION
	//----------------
	fixedp offset (32, 0, STEP);
	//if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
	//----------------

	vector<fixedp>				kseedsl (numk, fixedp (16, 16, 0)); for (int i=0; i<numk; i++)	kseedsl[i].convertTo (kseedsl_arg[i]);
	vector<fixedp>				kseedsa (numk, fixedp (16, 16, 0)); for (int i=0; i<numk; i++)	kseedsa[i].convertTo (kseedsa_arg[i]);
	vector<fixedp>				kseedsb (numk, fixedp (16, 16, 0)); for (int i=0; i<numk; i++)	kseedsb[i].convertTo (kseedsb_arg[i]);
	vector<fixedp>				kseedsx (numk, fixedp (16, 16, 0)); for (int i=0; i<numk; i++)	kseedsx[i].convertTo (kseedsx_arg[i]);
	vector<fixedp>				kseedsy (numk, fixedp (16, 16, 0)); for (int i=0; i<numk; i++)	kseedsy[i].convertTo (kseedsy_arg[i]);

	fixedp one (2, 0, float(1));
	fixedp zero (2, 0, float(0));

	vector<fixedp> clustersize (numk, fixedp (16, 16, 0));
	vector<fixedp> inv (numk, fixedp (16, 16, 0)); //to store 1/clustersize[k] values

	vector<fixedp> sigmal (numk, fixedp (16, 16, 0));
	vector<fixedp> sigmaa (numk, fixedp (16, 16, 0));
	vector<fixedp> sigmab (numk, fixedp (16, 16, 0));
	vector<fixedp> sigmax (numk, fixedp (16, 16, 0));
	vector<fixedp> sigmay (numk, fixedp (16, 16, 0));
	vector<fixedp> distvec (numk, fixedp (32, 0, FLT_MAX));

	fixedp invwt (2, 16, float(1.0/((STEP/M)*(STEP/M))));

	fixedp l(16,16,0), a(16,16,0), b(16,16,0);
	fixedp dist(8,8,0);
	fixedp distxy(8,8,0);

	fixedp l_diff(16,16,0), a_diff(16,16,0), b_diff(16,16,0), x_diff(16,16,0), y_diff(16,16,0);
	fixedp l_diff_sq(16,16,0), a_diff_sq(16,16,0), b_diff_sq(16,16,0), x_diff_sq(16,16,0), y_diff_sq(16,16,0);
	fixedp temp_dist(16,16,0);

	fixedp y1_limit(16,1,0), y2_limit(16,1,0), x1_limit(16,1,0), x2_limit(16,1,0);

	fixedp temp_l(16,16,0), temp_a(16,16,0), temp_b(16,16,0), temp_x(16,0,0), temp_y(16,0,0);
	fixedp temp_x_2(16,0,0), temp_y_2(16,0,0);
#else
	//----------------
	int offset = STEP;
	//if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
	//----------------

	float one = 1;
	float zero = 0;

	vector<float> clustersize(numk, 0);
	vector<Pixel> sigma(numk);

	vector<float> distvec(sz, DBL_MAX);

	float inv;
	float invwt = 1.0/((STEP/args.compactness)*(STEP/args.compactness));

	float l, a, b;
	float dist;
	float distxy;

	float l_diff, a_diff, b_diff, x_diff, y_diff;
	float l_diff_sq, a_diff_sq, b_diff_sq, x_diff_sq, y_diff_sq;
	float temp_dist;

	float y1_limit, y2_limit, x1_limit, x2_limit;

	float temp_l, temp_a, temp_b, temp_x, temp_y;
	int temp_x_2, temp_y_2;
#endif

	int x1, y1, x2, y2;

	for( int itr = 0; itr < args.iterations; itr++ )
	{
		ImageRasterScan image_scan (args.access_pattern[itr]);

#ifdef APPROXIMATION
		assign_fixed_vector (distvec, sz, FLT_MAX);
#else
		distvec.assign(sz, DBL_MAX);
#endif
		for( int n = 0; n < numk; n++ )
		{
			y1_limit = state.cluster_centers[n].y-offset;
			y2_limit = state.cluster_centers[n].y+offset;
			x1_limit = state.cluster_centers[n].x-offset;
			x2_limit = state.cluster_centers[n].x+offset;

			y1 = max(0.0f,				float(y1_limit));
			y2 = min((float)img.height,	float(y2_limit));
			x1 = max(0.0f,				float(x1_limit));
			x2 = min((float)img.width,	float(x2_limit));

			for( int y = y1; y < y2; y++ )
			{
				for( int x = x1; x < x2; x++ )
				{
					int i = y*img.width + x;

					if (!image_scan.is_exact_index (i))
						continue;

#ifdef APPROXIMATION
					l.convertTo (m_lvec[i]);
					a.convertTo (m_avec[i]);
					b.convertTo (m_bvec[i]);

					temp_x.convertTo (kseedsx[n]);
					temp_y.convertTo (kseedsy[n]);

					temp_x_2.convertTo (x);
					temp_y_2.convertTo (y);
#else
					l = img.data[i].l;
					a = img.data[i].a;
					b = img.data[i].b;

					temp_x = state.cluster_centers[n].x;
					temp_y = state.cluster_centers[n].y;

					temp_x_2 = x;
					temp_y_2 = y;
#endif

					l_diff = (l - state.cluster_centers[n].l);
					a_diff = (a - state.cluster_centers[n].a);
					b_diff = (b - state.cluster_centers[n].b);

					l_diff_sq = l_diff * l_diff;
					a_diff_sq = a_diff * a_diff;
					b_diff_sq = b_diff * b_diff;

					dist =			l_diff_sq + a_diff_sq + b_diff_sq;

					x_diff = (temp_x_2 - temp_x);
					y_diff = (temp_y_2 - temp_y);

					x_diff_sq = x_diff * x_diff;
					y_diff_sq = y_diff * y_diff;

					distxy =		x_diff_sq + y_diff_sq;

					//------------------------------------------------------------------------
					temp_dist = distxy*invwt;
					dist = dist + temp_dist;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
					//------------------------------------------------------------------------
					if( dist < distvec[i] )
					{
						distvec[i] = dist;
						state.labels[i]  = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		//instead of reassigning memory on each iteration, just reset.
#ifdef APPROXIMATION
		assign_fixed_vector (sigmal, numk, 0);
		assign_fixed_vector (sigmaa, numk, 0);
		assign_fixed_vector (sigmab, numk, 0);
		assign_fixed_vector (sigmax, numk, 0);
		assign_fixed_vector (sigmay, numk, 0);
		assign_fixed_vector (clustersize, numk, 0);
#else
		sigma.assign(numk, Pixel ());
		clustersize.assign(numk, zero);
#endif

		{int ind(0);
		for( int r = 0; r < img.height; r++ )
		{
			for( int c = 0; c < img.width; c++ )
			{
				// If not fitting the pyramid scan pattern or the pixel has not
				// been assigned to a SP yet, just continue
				if (!image_scan.is_exact_index (ind) || state.labels[ind] == -1)
				{
					ind++;
					continue;
				}

#ifdef APPROXIMATION
				temp_l.convertTo (m_lvec[c + m_width*r]);
				temp_a.convertTo (m_avec[c + m_width*r]);
				temp_b.convertTo (m_bvec[c + m_width*r]);
				temp_x.convertTo (c);
				temp_y.convertTo (r);

#else
				Pixel temp (
						img.data[c + img.width*r].l,
						img.data[c + img.width*r].a,
						img.data[c + img.width*r].b,
						c,
						r);
#endif
				sigma[state.labels[ind]] = sigma[state.labels[ind]] + temp;
				clustersize[state.labels[ind]] = clustersize[state.labels[ind]] + one;
				ind++;
			}
		}}

		{for( int k = 0; k < numk; k++ )
		{
			if( clustersize[k] <= zero ) clustersize[k] = 1;

			inv = one/clustersize[k];//computing inverse now to multiply, than divide later

			state.cluster_centers[k] = sigma[k] * inv;
		}}
	}

}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
int SLIC::EnforceLabelConnectivity()
{
	const int K = args.numlabels;  //the number of superpixels desired by the user
	int numlabels;	//the number of labels changes in the end if segments are removed

	int width = img.width;
	int height = img.height;
	vector<int> existing_labels = state.labels;

	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/K;
	state.labels.assign(sz, -1);
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > state.labels[oindex] )
			{
				state.labels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(state.labels[nindex] >= 0) adjlabel = state.labels[nindex];
					}
				}}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > state.labels[nindex] && existing_labels[oindex] == existing_labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								state.labels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						state.labels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;

	return numlabels;
}

