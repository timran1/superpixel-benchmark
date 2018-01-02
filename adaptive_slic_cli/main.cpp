
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include <bitset>

#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"

#include "adaptive_slic.h"
#include "helpers.h"

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
		("stateful", "Use state from previous frame")
		("tile-size", boost::program_options::value<int>()->default_value(0), "Size of side of tile square")
	    ("pyramid-pattern", boost::program_options::value<std::string>(), "access pattern for reverse pyramid approach")
	    ("target-error", boost::program_options::value<double>()->default_value(0.0), "target error to attempt, unless num iterations complete first.")
	    ("plot", "Plot iteration and error graphs.")
	    ("grid", "Draw SP grid.")
        ("parallel", "Parallize Algorithm.")
        ("gpu", "Parallelize on GPU.");
 
        
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

    std::vector<int> access_pattern;
    if (parameters.find("pyramid-pattern") != parameters.end()) {
    	std::string pattern_str = parameters["pyramid-pattern"].as<std::string>();
    	std::vector<std::string> tokens;

    	split(pattern_str, '*', std::back_inserter(tokens));
    	for (int i=0; i<tokens.size (); i++)
    	{
    		access_pattern.push_back (atoi (tokens[i].c_str ()));
    	}

    	access_pattern.back () = 1;
    }

    bool plot_graphs = false;
    if (parameters.find("plot") != parameters.end()) {
    	plot_graphs = true;
    }

    bool draw_grid = false;
    if (parameters.find("grid") != parameters.end()) {
        draw_grid = true;
    }
    

    
    bool parallel = false;
    if (parameters.find("parallel") != parameters.end()) {
    	parallel = true;
    }

    bool gpu = false;
    if (parameters.find("gpu") != parameters.end()) {
        gpu = true;
        parallel = true;
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
    int tile_size = parameters["tile-size"].as<int> ();
    double target_error = parameters["target-error"].as<double>();

    // Load AdaptiveSlic arguments.
    AdaptiveSlicArgs args;
    args.iterations = iterations;
    args.compactness = compactness;
    args.perturbseeds = perturb_seeds;
    args.color = color_space;
    args.stateful = stateful;
    args.numlabels = superpixels;
    args.tile_square_side = tile_size;
    args.target_error = target_error;

    if (access_pattern.size() > 0)
    {
    	args.access_pattern = access_pattern;
    	if (args.access_pattern.size () != args.iterations)
    	{
    		args.access_pattern.resize (args.iterations, 1);
    	}
    }else
    {
        args.access_pattern.resize (iterations, 1);
    }


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

        int target_error_factor = 1000;

        cv::namedWindow(window_name, 0);
        cv::createTrackbar("Superpixels", window_name, &superpixels, 10000, 0);
        cv::createTrackbar("Iterations", window_name, &iterations, 12, 0);
        cv::createTrackbar("Target Error", window_name, &target_error_factor, 1000, 0);

        AdaptiveSlic adaptive_slic (plot_graphs);
        for (;;)
        {
            cv::Mat image, mask;

            // Update args for this frame.
            if (!args.stateful ||
					args.iterations != iterations ||
					args.numlabels != superpixels)
            {
            	adaptive_slic.reset ();
            }

            if (args.iterations != iterations)
            	args.access_pattern.resize (iterations, 1);

            args.iterations = iterations;
            args.numlabels = superpixels;
            args.target_error = float (target_error_factor)/10000;
            args.parallel = parallel;
            args.gpu = gpu;

            // Capture a frame.
            cap >> image;

            if( image.empty() )
            {
                std::cout << "Empty frame received. Exiting..." << std::endl;
                exit (1);
            }

            boost::timer timer;

            adaptive_slic.compute_superpixels(image, args);

            float elapsed = timer.elapsed();

            // Create grid of SP association
            if (draw_grid)
                image.setTo(cv::Scalar(255, 0, 0), adaptive_slic.grid_mat);

            // Create contours for display.
            ImageUtils::get_labels_contour_mask(image, adaptive_slic.get_labels (), mask, (superpixels < 1000));
            image.setTo(cv::Scalar(0, 0, 255), mask);

            float fps = 1/elapsed;
            std::stringstream ss;
            ss << fps << " fps";
            putText(image, ss.str (), cvPoint(5,10),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 0.2, CV_AA);

            cv::imshow(window_name, image);

            cout << adaptive_slic.profiler.print_checkpoints ();

			float display = timer.elapsed();
            std::cout << "Size: " << image.cols << "x" << image.rows << " Clust. Upd: " << args.num_clusters_updated <<" - Processing Time: " << elapsed << "s - Including display: " << display << "s" << std::endl;

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

            args.stateful = false;	// No statefulness needed for pictures

            boost::timer timer;
            AdaptiveSlic adaptive_slic (false);
            adaptive_slic.compute_superpixels(image, args);
            float elapsed = timer.elapsed();
            total += elapsed;

            cout << adaptive_slic.profiler.print_checkpoints ();

            std::cout << "Size: " << image.cols << "x" << image.rows << " Clust. Upd: " << args.num_clusters_updated <<" - Processing Time: " << elapsed << "s" << std::endl;

            if (!output_dir.empty()) {
                boost::filesystem::path csv_file(output_dir
                        / boost::filesystem::path(prefix + it->second.stem().string() + ".csv"));
                IOUtil::writeMatCSV<int>(csv_file, adaptive_slic.get_labels ());
            }

            if (!vis_dir.empty()) {
                boost::filesystem::path contours_file(vis_dir
                        / boost::filesystem::path(prefix + it->second.stem().string() + ".png"));
                cv::Mat image_contours;
                Visualization::drawContours(image, adaptive_slic.get_labels (), image_contours);
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
