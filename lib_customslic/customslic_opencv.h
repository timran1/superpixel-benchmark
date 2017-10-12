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

#ifndef CUSTOMSLIC_OPENCV_H
#define	CUSTOMSLIC_OPENCV_H

#include <opencv2/opencv.hpp>

#include "SLIC.h"
#include "plotter.h"

/** \brief Wrapper for running SLIC on OpenCV images.
 * \author David Stutz
 */
class CUSTOMSLIC_OpenCV {
public:
	CUSTOMSLIC_OpenCV (bool plot);

    void computeSuperpixels_extended(const cv::Mat image, CUSTOMSLIC_ARGS& args);

    void reset ();

    cv::Mat get_labels ();

    // Image helper functions. Maybe these can be moved somewhere else.
    static void getLabelContourMask(const cv::Mat mat,
            cv::Mat labels, cv::OutputArray _mask, bool _thick_line);

public:
    void computeSuperpixels(shared_ptr<SLIC>, cv::Mat &labels_seg, CUSTOMSLIC_ARGS& args);
    vector<shared_ptr<SLIC>> slics;
    Plotter plotter;
    bool plot;

private:
    cv::Mat labels;
	cv::Mat image_mat;
    cv::Rect labels_rect;
};

#endif	/* SLIC_OPENCV_H */

