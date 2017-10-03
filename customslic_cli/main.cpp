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

#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include <bitset>
#include "customslic_opencv.h"
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

/** \brief Command line tool for running customslic.
 * Usage:
 * \code{sh}
 *   $ ../bin/customslic_cli --help
 *   Allowed options:
 *     -h [ --help ]                   produce help message
 *     -i [ --input ] arg              the folder to process (can also be passed as 
 *                                     positional argument)
 *     -s [ --superpixels ] arg (=400) number of superpixles
 *     -c [ --compactness ] arg (=40)  compactness
 *     -p [ --perturb-seeds ] arg (=1) perturb seeds: > 0 yes, = 0 no
 *     -t [ --iterations ] arg (=10)   iterations
 *     -r [ --color-space ] arg (=1)   color space: 0 = RGB, > 0 = Lab
 *     -o [ --csv ] arg                specify the output directory (default is 
 *                                     ./output)
 *     -v [ --vis ] arg                visualize contours
 *     -x [ --prefix ] arg             output file prefix
 *     -w [ --wordy ]                  verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main(int argc, const char** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", boost::program_options::value<std::string>(), "the folder to process (can also be passed as positional argument)")
        ("superpixels,s", boost::program_options::value<int>()->default_value(400), "number of superpixles")
        ("compactness,c", boost::program_options::value<double>()->default_value(40.), "compactness")
        ("perturb-seeds,p", boost::program_options::value<int>()->default_value(1), "perturb seeds: > 0 yes, = 0 no")
        ("iterations,t", boost::program_options::value<int>()->default_value(10), "iterations")
        ("color-space,r", boost::program_options::value<int>()->default_value(1), "color space: 0 = RGB, > 0 = Lab")
        ("csv,o", boost::program_options::value<std::string>()->default_value(""), "specify the output directory (default is ./output)")
        ("vis,v", boost::program_options::value<std::string>()->default_value(""), "visualize contours")
        ("prefix,x", boost::program_options::value<std::string>()->default_value(""), "output file prefix")
		("video-file", boost::program_options::value<std::string>(), "path to video file")
        ("wordy,w", "verbose/wordy/debug")
		("camera", "Use camera feed")
		("small-video", "Set camera frame size = 320x240")
		("stateful", "Use state from previous frame");
        
    boost::program_options::positional_options_description positionals;
    positionals.add("input", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path output_dir(parameters["csv"].as<std::string>());
    if (!output_dir.empty()) {
        if (!boost::filesystem::is_directory(output_dir)) {
            boost::filesystem::create_directories(output_dir);
        }
    }
    
    boost::filesystem::path vis_dir(parameters["vis"].as<std::string>());
    if (!vis_dir.empty()) {
        if (!boost::filesystem::is_directory(vis_dir)) {
            boost::filesystem::create_directories(vis_dir);
        }
    }
    
    bool use_camera = false;
    if (parameters.find("camera") != parameters.end()) {
        // We want a camera feed.
        use_camera = true;
    }

    bool use_video_file = false;
    if (parameters.find("video-file") != parameters.end()) {
        // We want a camera feed.
    	use_video_file = true;
    }

    bool small_video = false;
    if (parameters.find("small-video") != parameters.end()) {
        // We want a camera feed.
        small_video = true;
    }

    bool stateful = false;
    if (parameters.find("stateful") != parameters.end()) {
        // We want a camera feed.
    	stateful = true;
    }


    bool wordy = false;
    if (parameters.find("wordy") != parameters.end()) {
        wordy = true;
    }
    
    int superpixels = parameters["superpixels"].as<int>();
    double compactness = parameters["compactness"].as<double>();
    int iterations = parameters["iterations"].as<int>();
    int perturb_seeds_int = parameters["perturb-seeds"].as<int>();
    bool perturb_seeds = perturb_seeds_int > 0 ? true : false;
    int color_space = parameters["color-space"].as<int> ();
    
    if (use_camera || use_video_file)
    {
        int capture = 0; // Camera ID

        cv::VideoCapture cap;
        if (use_video_file)
        {
        	boost::filesystem::path video_file(parameters["video-file"].as<std::string>());
        	std::cout << "Opening file: " << video_file.string() << std::endl;
        	cap = cv::VideoCapture (video_file.string());
        }
        else if( !cap.open(capture) )
        {
            std::cout << "Could not initialize capturing..." << capture << std::endl;
            return -1;
        }

        if (small_video)
        {
            cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
        }

        const char* window_name = "SLIC Superpixels";

        cv::namedWindow(window_name, 0);
        cv::createTrackbar("Superpixels", window_name, &superpixels, 10000, 0);
        cv::createTrackbar("Iterations", window_name, &iterations, 12, 0);

        cv::Mat result, mask;

        for (;;)
        {
            cv::Mat image;
            cv::Mat labels;

            // Capture a frame.
            cap >> image;

            if( image.empty() )
            {
                std::cout << "Empty frame received. Exiting..." << std::endl;
            }

            result = image;

            CUSTOMSLIC_ARGS args;
            args.region_size = SuperpixelTools::computeRegionSizeFromSuperpixels(image, superpixels);
            args.iterations = iterations;
            args.compactness = compactness;
            args.perturbseeds = perturb_seeds;
            args.color = color_space;
            args.stateful = stateful;
            args.numlabels = superpixels;

            boost::timer timer;
            CUSTOMSLIC_OpenCV::computeSuperpixels_extended(image, labels, args);
            float elapsed = timer.elapsed();

            // Create contours for display.
            CUSTOMSLIC_OpenCV::getLabelContourMask(image, labels, mask, (superpixels < 1000));
            result.setTo(cv::Scalar(0, 0, 255), mask);

            cv::imshow(window_name, result);

			float display = timer.elapsed();
            std::cout << "Size: " << result.cols << "x" << result.rows << " - Time: " << elapsed << "s - Display: " << display << "s" << std::endl;

			// Wait for some inputs and prepare for next time.
            int c = cv::waitKey(1) & 0xff;
            if( c == 'q' || c == 'Q' || c == 27 )
            {
                std::cout << "Exiting on user input..." << std::endl;
                break;
            }
        }
    }
    else
    {
		boost::filesystem::path input_dir(parameters["input"].as<std::string>());
		if (!boost::filesystem::is_directory(input_dir)) {
			std::cout << "Image directory not found ..." << std::endl;
			return 1;
		}

	    std::string prefix = parameters["prefix"].as<std::string>();

        std::multimap<std::string, boost::filesystem::path> images;
        std::vector<std::string> extensions;
        IOUtil::getImageExtensions(extensions);
        IOUtil::readDirectory(input_dir, extensions, images);
        
        float total = 0;
        for (std::multimap<std::string, boost::filesystem::path>::iterator it = images.begin();
                it != images.end(); ++it) {

            cv::Mat image = cv::imread(it->first);
            cv::Mat labels;

            CUSTOMSLIC_ARGS args;
            args.region_size = SuperpixelTools::computeRegionSizeFromSuperpixels(image, superpixels);
            args.iterations = iterations;
            args.compactness = compactness;
            args.perturbseeds = perturb_seeds;
            args.color = color_space;
            args.stateful = false;
            args.numlabels = superpixels;

            boost::timer timer;
            CUSTOMSLIC_OpenCV::computeSuperpixels_extended(image, labels, args);
            float elapsed = timer.elapsed();
            total += elapsed;

            if (!output_dir.empty()) {
                boost::filesystem::path csv_file(output_dir
                        / boost::filesystem::path(prefix + it->second.stem().string() + ".csv"));
                IOUtil::writeMatCSV<int>(csv_file, labels);
            }

            if (!vis_dir.empty()) {
                boost::filesystem::path contours_file(vis_dir
                        / boost::filesystem::path(prefix + it->second.stem().string() + ".png"));
                cv::Mat image_contours;
                Visualization::drawContours(image, labels, image_contours);
                cv::imwrite(contours_file.string(), image_contours);
            }
        }
        
        if (wordy) {
            std::cout << "Average time: " << total / images.size() << "." << std::endl;
        }
        
        if (!output_dir.empty()) {
            std::ofstream runtime_file(output_dir.string() + "/" + prefix + "runtime.txt",
                    std::ofstream::out | std::ofstream::app);

            runtime_file << total / images.size() << "\n";
            runtime_file.close();
        }
    }
    
    return 0;
}
