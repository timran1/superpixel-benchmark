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

void CUSTOMSLIC_OpenCV::computeSuperpixels(SLIC &slic, const cv::Mat &mat, cv::Mat &labels, CUSTOMSLIC_ARGS& args) {
    
    // Convert matrix to unsigned int array.
    unsigned int* image = new unsigned int[mat.rows*mat.cols];
    unsigned int value = 0x0000;

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {

            int b = mat.at<cv::Vec3b>(i,j)[0];
            int g = mat.at<cv::Vec3b>(i,j)[1];
            int r = mat.at<cv::Vec3b>(i,j)[2];

            value = 0x0000;
            value |= (0xFF000000);
            value |= (0x00FF0000 & (r << 16));
            value |= (0x0000FF00 & (g << 8));
            value |= (0x000000FF & b);

            image[j + mat.cols*i] = value;
        }
    }

    int* segmentation = new int[mat.rows*mat.cols];
    int number_of_labels_desired = args.numlabels;

    slic.DoSuperpixelSegmentation_ForGivenSuperpixelStep(image, mat.cols, 
            mat.rows, segmentation, args);

    // Convert labels.
    labels.create(mat.rows, mat.cols, CV_32SC1);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            labels.at<int>(i, j) = segmentation[j + i*mat.cols];
        }
    }
}

void CUSTOMSLIC_OpenCV::EnforceLabelConnectivity_extended (
		SLIC &slic,
		cv::Mat &labels,
		CUSTOMSLIC_ARGS& args
		)
{
	int sz = labels.cols*labels.rows;
	int* segmentation = new int[sz];

	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			segmentation[j + i*labels.cols] = labels.at<int>(i, j);
		}
	}

	int number_of_labels_desired = args.numlabels;

	slic.EnforceLabelConnectivity_extended(segmentation, labels.cols,
			labels.rows, number_of_labels_desired, args.numlabels);

	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			labels.at<int>(i, j) = segmentation[j + i*labels.cols];
		}
	}

	if(segmentation) delete [] segmentation;

}


void CUSTOMSLIC_OpenCV::computeSuperpixels_extended(const cv::Mat &mat, cv::Mat &labels, CUSTOMSLIC_ARGS& args) {
    
    bool tiling = args.tile_square_side > 0;

    int unconnected_components = 0;

    if (!tiling)
    {
        SLIC slic;

        computeSuperpixels(slic, mat, labels, args);

        EnforceLabelConnectivity_extended (slic, labels, args);

        unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels);
    }
    else
    {
    	args.stateful = false;

        // tiling variables
        int square_side = 64;

        // pad image and labels to make full squares
        cv::Mat padded_mat;
        int padding_c = square_side - (mat.cols % square_side);
        int padding_r = square_side - (mat.rows % square_side);
        padded_mat.create(mat.rows + padding_r, mat.cols + padding_c, mat.type());
        padded_mat.setTo(cv::Scalar::all(0));
        mat.copyTo (padded_mat (Rect (0, 0, mat.cols, mat.rows)));

        cv::Mat padded_labels (padded_mat.rows, padded_mat.cols, CV_32SC1);

        //imshow( "padded_labels", padded_mat ); //cv::waitKey(0);
        vector<cv::Mat> img_segs;
        vector<cv::Mat> labels_segs;

        // create square segments
        for (int r = 0; r <= (padded_mat.rows - square_side); r += square_side)
        {
            for (int c = 0; c <= (padded_mat.cols - square_side); c += square_side)
            {
                cv::Mat region (square_side, square_side, padded_mat.type());
                region = padded_mat (Rect (c, r, square_side, square_side));

                img_segs.push_back (region);
                labels_segs.push_back (cv::Mat ());
                
                //imshow( "img_segs", region ); cv::waitKey(0);
            }
        }

        int num_segments = img_segs.size ();
        int num_sp_per_segment = args.numlabels / num_segments;

        int num_sp_so_far = 0;
        for (int i = 0; i< num_segments; i++)
        {
            // compute region size for size of new array
            args.region_size = SuperpixelTools::computeRegionSizeFromSuperpixels(img_segs[i],
                            num_sp_per_segment);

            SLIC slic;

            // the main super pixel algo
            computeSuperpixels(slic, img_segs[i], labels_segs[i], args);

            // TODO: Check if this can be moved out of the loop.
            int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels_segs[i]);

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
        labels = padded_labels (Rect (0, 0, mat.cols, mat.rows));
    }

}

void CUSTOMSLIC_OpenCV::getLabelContourMask(const cv::Mat &mat, cv::Mat &labels, cv::OutputArray _mask, bool _thick_line)
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
