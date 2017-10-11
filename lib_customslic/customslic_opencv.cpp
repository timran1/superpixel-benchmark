/**
 * Copyright (c) 2016, David Stutz
 * Contact: david.stutz@rwth-aachen.de, davidstutz.de
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "SLIC.h"
#include "customslic_opencv.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "superpixel_tools.h"

#include "stdio.h"
using namespace cv;

CUSTOMSLIC_OpenCV::CUSTOMSLIC_OpenCV (bool plot) : plotter (10)
{
	this->plot = plot;
	if (plot)
	{
		// Set the plot
		plotter.pre_plot_cmds =
		"set multiplot layout 2, 1 title \"Adaptive SLIC\" font \",14\" \n"
		"set tmargin 3 \n";
		plotter.post_plot_cmds = "unset multiplot \n";

		// Error plot
		plotter.plots["plot1"] = Plot ("Error - avg cluster movement (per cluster per search area pixel)");
		plotter.plots["plot1"].pre_plot_cmds = "set key left bottom \n";
		plotter.plots["plot1"].series["error"] = DataSeries (" ", "Error", "line", 500);
		plotter.plots["plot1"].series["target"] = DataSeries (" ", "Target", "line", 500);

		// Iterations plot
		plotter.plots["plot2"] = Plot ("Iterations");
		plotter.plots["plot2"].pre_plot_cmds = "unset key\n";
		plotter.plots["plot2"].series["iter"] = DataSeries ("[ ] [0:]", "Iterations", "boxes fs solid ", 500);
	}
}

void CUSTOMSLIC_OpenCV::reset ()
{
	slics.clear ();
	imgs.clear ();
}

void CUSTOMSLIC_OpenCV::computeSuperpixels(shared_ptr<SLIC> slic, cv::Mat &labels, CUSTOMSLIC_ARGS& args) {

	// Main operation
	slic->initIterationState ();

	int itr = 0;
	for(; itr < args.iterations; itr++ )
	{
		slic->PerformSuperpixelSLICIteration(itr);

		// post iteration hook.
		if (slic->iter_state.iteration_error > args.target_error)
			break;
	}

    // Post processing
    slic->EnforceLabelConnectivity ();

    slic->GetLabelsMat (labels);

    SuperpixelTools::relabelConnectedSuperpixels(labels);

    // post tile SLIC hook.
}

void showMat (cv::Mat mat, std::string label)
{
    double min_val, max_val;
    cv::minMaxIdx(mat, &min_val, &max_val);
    cv::Mat adjMap;
    cv::convertScaleAbs(mat, adjMap, 255 / max_val);
    cv::imshow(label.c_str (), adjMap); cv::waitKey(0);
}

void CUSTOMSLIC_OpenCV::computeSuperpixels_extended(const cv::Mat mat_rgb, cv::Mat &labels, CUSTOMSLIC_ARGS& args) {
    
    bool tiling = args.tile_square_side > 0;

    // Convert image to CIE LAB color space.
    cv::Mat mat_lab = SLIC::DoRGBtoLABConversion (mat_rgb);

    if (!tiling)
    {

        // Convert image to format suitable for SLIC algo.
    	imgs.push_back (make_shared<Image> (mat_lab, mat_lab.cols, mat_lab.rows));

    	// Make a new SLIC segmentor if this is first call to this function.
        if (slics.size () == 0)
        {
        	slics.push_back (make_shared<SLIC> (imgs.back (), args));

        	// set initial seeds
        	slics.back ()->SetInitialSeeds ();
        }
        else
        	// Update reference to new image for existing SLICs
        	slics.back ()->setImage (imgs.back ());

        computeSuperpixels(slics.back (), labels, args);

		if (plot)
		{
			auto slic = slics.back ();
			plotter.plots["plot1"].series["error"].add_point (slic->iter_state.iteration_error);
			plotter.plots["plot2"].series["iter"].add_point (slic->iter_state.iter_num);
		}
    }
    else
    {

        int square_side = args.tile_square_side;

    	args.one_sided_padding = false;

    	// Determine amount of padding required.
        cv::Mat padded_mat;
        int padding_c = square_side - (mat_lab.cols % square_side);
        int padding_c_left = args.one_sided_padding ? 0 : padding_c / 2;
        int padding_r = square_side - (mat_lab.rows % square_side);
        int padding_r_up = args.one_sided_padding ? 0 : padding_r / 2;

        // pad image and labels to make full squares
        padded_mat.create(mat_lab.rows + padding_r, mat_lab.cols + padding_c, mat_lab.type());
        padded_mat.setTo(cv::Scalar::all(0));
        mat_lab.copyTo (padded_mat (Rect (padding_c_left, padding_r_up, mat_lab.cols, mat_lab.rows)));

        cv::Mat padded_labels (padded_mat.rows, padded_mat.cols, CV_32SC1);

        //imshow( "padded_labels", padded_mat ); //cv::waitKey(0);
        vector<cv::Mat> mat_segs;
        vector<cv::Mat> labels_segs;

        // create square segments
        for (int r = 0; r <= (padded_mat.rows - square_side); r += square_side)
        {
            for (int c = 0; c <= (padded_mat.cols - square_side); c += square_side)
            {
                cv::Mat region (square_side, square_side, padded_mat.type());
                region = padded_mat (Rect (c, r, square_side, square_side));

                mat_segs.push_back (region);
                labels_segs.push_back (cv::Mat ());
                imgs.push_back (make_shared<Image> (mat_segs.back (), mat_segs.back ().cols, mat_segs.back ().rows));
            }
        }

        int num_segments = imgs.size ();

        // We do not need to increase number of superpixels to account for padding.
        // This is because args.region_size is same. args.region size is used by
        // SLIC::SetInitialSeeds to determine number of SPs to create and places
        // initial seeds accordingly.

    	// Make new SLIC segmentors if this is first call to this function.
        if (slics.size () == 0)
        {
			for (int i = 0; i< num_segments; i++)
			{
				slics.push_back (make_shared<SLIC> (imgs[i], args));

				// This is the *expected* region size of each SP.
				slics.back ()->state.updateRegionSizeFromSuperpixels (mat_lab.rows * mat_lab.cols, args.numlabels);

            	// set initial seeds
				slics.back ()->SetInitialSeeds ();
			}
        }
        else
        {
        	// Update reference to new image for existing SLICs
			for (int i = 0; i< num_segments; i++)
				slics[i]->setImage (imgs[i]);
        }

        int num_sp_so_far = 0;
        for (int i=0; i<num_segments; i++)
        {
        	if (slics[i]->state.cluster_centers.size () > 0)
				computeSuperpixels(slics[i], labels_segs[i], args);
        	else
        	{
        		// Scan region is bigger than the tile size. Cannot run algo, simply
        		// return the whole tile as a single SP.
        		labels_segs[i].create(mat_segs[i].rows, mat_segs[i].cols, CV_32SC1);
        		labels_segs[i].setTo(cv::Scalar::all(i));
        	}

            //showMat (mat_segs[i], "labels");
            //showMat (labels_segs[i], "labels");

            // Increment the label numbers with the number of SPs generated so far.
            int num_sp_generated = SuperpixelTools::countSuperpixels(labels_segs[i]);
            labels_segs[i] += Scalar(num_sp_so_far);
            num_sp_so_far += num_sp_generated;
        }

        // join label segments
        int i = 0;
        for (int r = 0; r <= (padded_mat.rows - square_side); r += square_side)
        {
            for (int c = 0; c <= (padded_mat.cols - square_side); c += square_side)
            {
                labels_segs[i].copyTo (padded_labels (Rect (c, r, square_side, square_side)));
                i++;
            }
        }
        // Extract out only the required region from labels
        labels = padded_labels (Rect (padding_c_left, padding_r_up, mat_lab.cols, mat_lab.rows));

        //showMat (labels, "labels");
        //printf ("num_sp_so_far at end=%d\n", num_sp_so_far);

        if (plot)
		{
			float total_error = 0;
			float total_iter = 0;
			for (auto& slic:slics)
			{
				total_error += slic->iter_state.iteration_error;
				total_iter += slic->iter_state.iter_num;
			}
			plotter.plots["plot1"].series["error"].add_point (total_error/slics.size ());
			plotter.plots["plot2"].series["iter"].add_point (total_iter/slics.size ());
		}
    }

    // post image SLIC hook.
    if (plot)
    {
    	plotter.plots["plot1"].series["target"].add_point (args.target_error);
    	plotter.do_plot ();
    }

    // Cleanups
    imgs.clear ();

}

void CUSTOMSLIC_OpenCV::getLabelContourMask(const cv::Mat mat, cv::Mat labels, cv::OutputArray _mask, bool _thick_line)
{
    // default width
    int line_width = 2;

    if ( !_thick_line ) line_width = 1;

    int m_width = mat.cols;
    int m_height = mat.rows;

    _mask.create( m_height, m_width, CV_8UC1 );
    Mat mask = _mask.getMat();

    mask.setTo(0);

    const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy8[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };

    int sz = m_width*m_height;

    vector<bool> istaken(sz, false);

    int mainindex = 0;
    for( int j = 0; j < m_height; j++ )
    {
      for( int k = 0; k < m_width; k++ )
      {
        int np = 0;
        for( int i = 0; i < 8; i++ )
        {
          int x = k + dx8[i];
          int y = j + dy8[i];

          if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
          {
            int index = y*m_width + x;

            if( false == istaken[index] )
            {
              if( labels.at<int>(j,k) != labels.at<int>(y,x) ) np++;
            }
          }
        }
        if( np > line_width )
        {
           mask.at<char>(j,k) = (uchar)255;
           istaken[mainindex] = true;
        }
        mainindex++;
      }
    }
}
