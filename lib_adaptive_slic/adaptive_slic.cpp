
#include "slic.h"
#include "adaptive_slic.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "superpixel_tools.h"

#include "stdio.h"
using namespace cv;

AdaptiveSlic::AdaptiveSlic (bool plot) : plotter (10)
{
    labels_rect = Rect (0,0,0,0);

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

void
AdaptiveSlic::reset ()
{
	slics.clear ();
	labels.release ();
    labels_rect = Rect (0,0,0,0);
	image_mat = cv::Mat ();
}

cv::Mat
AdaptiveSlic::get_labels ()
{
    // Extract out only the required region from labels
    return labels (labels_rect);
}

void
AdaptiveSlic::compute_superpixels_on_tile (shared_ptr<SLIC> slic, cv::Mat &labels_seg, AdaptiveSlicArgs& args)
{
	// Main operation
	slic->init_iteration_state ();

	int itr = 0;
	for(; itr < args.iterations; itr++ )
	{
		slic->perform_superpixel_slic_iteration ();

		// post iteration hook.
		if (slic->iter_state.iteration_error > args.target_error)
			break;
	}

    // Post processing
    slic->enforce_labels_connectivity ();

    slic->get_labels_mat (labels_seg);

    // Rename labels so they consist of continuously increasing
    // numbers.
    SuperpixelTools::relabelSuperpixels(labels_seg);

}


void
AdaptiveSlic::compute_superpixels (const cv::Mat mat_rgb, AdaptiveSlicArgs& args)
{
    bool tiling = args.tile_square_side > 0;

    if (!tiling)
    {
    	// Create Mat to hold CIELAB format image.
    	if (image_mat.rows != mat_rgb.rows)
    		image_mat = cv::Mat::zeros(mat_rgb.rows, mat_rgb.cols, CV_32FC3);

        // Convert image to CIE LAB color space.
    	ImageUtils::do_rgb_to_lab_conversion (mat_rgb, image_mat, 0, 0);

        // Convert image to format suitable for SLIC algo.
        vector<shared_ptr<Image>> imgs;
    	imgs.push_back (make_shared<Image> (image_mat, image_mat.cols, image_mat.rows));

    	// Create labels array to hold labels info
    	if (labels.rows != image_mat.rows)
    	{
    		labels.create (image_mat.rows, image_mat.cols, CV_32SC1);
		    labels_rect = Rect (0, 0, image_mat.cols, image_mat.rows);
    	}

    	// Make a new SLIC segmentor if this is first call to this function.
        if (slics.size () == 0)
        {
        	slics.push_back (make_shared<SLIC> (imgs.back (), args));

        	// set initial seeds
        	slics.back ()->set_initial_seeds ();
        }
        else
        	// Update reference to new image for existing SLICs
        	slics.back ()->set_image (imgs.back ());

        compute_superpixels_on_tile (slics.back (), labels, args);

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
    	int padding_c = square_side - (mat_rgb.cols % square_side);
    	int padding_c_left = args.one_sided_padding ? 0 : padding_c / 2;
		int padding_r = square_side - (mat_rgb.rows % square_side);
		int padding_r_up = args.one_sided_padding ? 0 : padding_r / 2;

    	// Create Mat to hold CIELAB format image.
    	if (image_mat.rows != mat_rgb.rows)
    		image_mat = cv::Mat::zeros(mat_rgb.rows + padding_r, mat_rgb.cols + padding_c, CV_32FC3);

		// Convert image to CIE LAB color space.
    	ImageUtils::do_rgb_to_lab_conversion (mat_rgb, image_mat, padding_c_left, padding_r_up);

		// Create labels array to hold labels info
		if (labels.rows != image_mat.rows)
		{
			labels.create (image_mat.rows, image_mat.cols, CV_32SC1);
		    labels_rect = Rect (padding_c_left, padding_r_up, mat_rgb.cols, mat_rgb.rows);
		}

        vector<cv::Mat> mat_segs;
        vector<cv::Mat> labels_segs;
        vector<shared_ptr<Image>> imgs;

        // create square segments
        for (int r = 0; r <= (image_mat.rows - square_side); r += square_side)
        {
            for (int c = 0; c <= (image_mat.cols - square_side); c += square_side)
            {
                // This only creates a reference to the bigger image (ROI)
				cv::Mat region = image_mat (Rect (c, r, square_side, square_side));
            	cv::Mat label_region = labels (Rect (c, r, square_side, square_side));

				mat_segs.push_back (region);
                labels_segs.push_back (label_region);
                imgs.push_back (make_shared<Image> (region, region.cols, region.rows));
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
				slics.back ()->state.update_region_size_from_sp (mat_rgb.rows * mat_rgb.cols, args.numlabels);

            	// set initial seeds
				slics.back ()->set_initial_seeds ();
			}
        }
        else
        {
        	// Update reference to new image for existing SLICs
			for (int i = 0; i< num_segments; i++)
				slics[i]->set_image (imgs[i]);
        }

        int num_sp_so_far = 0;
        for (int i=0; i<num_segments; i++)
        {
        	if (slics[i]->state.cluster_centers.size () > 0)
        		compute_superpixels_on_tile (slics[i], labels_segs[i], args);
        	else
        	{
        		// Scan region is bigger than the tile size. Cannot run algo, simply
        		// return the whole tile as a single SP.
        		labels_segs[i].setTo(cv::Scalar::all(i));
        	}

            //ImageUtils::show_mat (mat_segs[i], "labels");
            //ImageUtils::show_mat (labels_segs[i], "labels");

            // Increment the label numbers with the number of SPs generated so far.
        	// Post processing can only decrease number of SPs, not increase it.
            int num_sp_generated = slics[i]->state.cluster_centers.size ();
            labels_segs[i] += Scalar(num_sp_so_far);
            num_sp_so_far += num_sp_generated;
        }

        //ImageUtils::show_mat (labels, "labels");

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

}

//----------------------------------------------------------------------------
//ImageUtils function definitions.
//----------------------------------------------------------------------------
void
ImageUtils::get_labels_contour_mask (const cv::Mat mat, cv::Mat labels, cv::OutputArray _mask, bool _thick_line)
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

void
ImageUtils::RGB2XYZ (const int sR, const int sG, const int sB, float& X, float& Y, float& Z)
{
	// sRGB (D65 illuninant assumption) to XYZ conversion
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

void
ImageUtils::RGB2LAB (const int sR, const int sG, const int sB, float& lval, float& aval, float& bval)
{
	// sRGB to XYZ conversion
	float X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	// XYZ to LAB conversion
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

void
ImageUtils::do_rgb_to_lab_conversion (const cv::Mat &mat, cv::Mat &out, int padding_c_left, int padding_r_up)
{
	assert (mat.rows+padding_r_up <= out.rows && mat.cols+padding_c_left <= out.cols);

	// Ranges:
	// L in [0, 100]
	// A in [-86.185, 98,254]
	// B in [-107.863, 94.482]

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {

            int b = mat.at<cv::Vec3b>(i,j)[0];
            int g = mat.at<cv::Vec3b>(i,j)[1];
            int r = mat.at<cv::Vec3b>(i,j)[2];

            int arr_index = j + mat.cols*i;

            float l_out, a_out, b_out;
    		RGB2LAB( r, g, b, l_out, a_out, b_out);

    		out.at<cv::Vec3f>(i+padding_r_up,j+padding_c_left)[0] = l_out;
    		out.at<cv::Vec3f>(i+padding_r_up,j+padding_c_left)[1] = a_out;
    		out.at<cv::Vec3f>(i+padding_r_up,j+padding_c_left)[2] = b_out;
        }
    }
}

void
ImageUtils::show_mat (cv::Mat mat, std::string label)
{
    double min_val, max_val;
    cv::minMaxIdx(mat, &min_val, &max_val);
    cv::Mat adjMap;
    cv::convertScaleAbs(mat, adjMap, 255 / max_val);
    cv::imshow(label.c_str (), adjMap); cv::waitKey(0);
}

