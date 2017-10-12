
#include <cmath>
#include <iostream>
#include <assert.h>

#include "slic.h"

using namespace std;

//-----------------------------------------------------------------------
// SLIC class.
//-----------------------------------------------------------------------
SLIC::SLIC(shared_ptr<Image> img, AdaptiveSlicArgs& args)
	: img (img), args (args)
{
	state.init (args, img->width * img->height);
}

void
SLIC::init_iteration_state ()
{
	iter_state.init (img->width * img->height, state.cluster_centers.size ());
}

void
SLIC::get_labels_mat (cv::Mat &labels)
{
	// Convert labels.
	for (int i = 0; i < img->height; ++i) {
		for (int j = 0; j < img->width; ++j) {
			labels.at<int>(i, j) = state.labels[j + i*img->width];
		}
	}
}

void
SLIC::set_initial_seeds ()
{
	define_image_pixels_association();

	if (args.perturbseeds)
	{
		vector<float> edges;
		detect_lab_edges (edges);
		perturb_seeds (edges);
	}

	// compute region size for number of clusters actually created. args.region_size is
    // used to determine search area during SLIC iterations.
	state.update_region_size_from_sp (img->width *img->height, state.cluster_centers.size ());
}

void
SLIC::detect_lab_edges (vector<float>& edges)
{
       int& width = img->width;
       int& height = img->height;

       int sz = width*height;

       edges.resize(sz,0);
       for( int j = 1; j < img->height-1; j++ )
       {
               for( int k = 1; k < img->width-1; k++ )
               {
				   int i = j*img->width+k;
				   float dx = (img->data[i-1].l ()-img->data[i+1].l ())*(img->data[i-1].l ()-img->data[i+1].l ()) +
								  (img->data[i-1].a ()-img->data[i+1].a ())*(img->data[i-1].a ()-img->data[i+1].a ()) +
								  (img->data[i-1].b ()-img->data[i+1].b ())*(img->data[i-1].b ()-img->data[i+1].b ());

				  float dy = (img->data[i-width].l ()-img->data[i+width].l ())*(img->data[i-width].l ()-img->data[i+width].l ()) +
								  (img->data[i-width].a ()-img->data[i+width].a ())*(img->data[i-width].a ()-img->data[i+width].a ()) +
								 (img->data[i-width].b ()-img->data[i+width].b ())*(img->data[i-width].b ()-img->data[i+width].b ());

				   edges[i] = dx*dx + dy*dy;
               }
       }
}

void
SLIC::perturb_seeds(const vector<float>& edges)
{
       vector<Pixel>& cluster_centers = state.cluster_centers;

       const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
       const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

       int numseeds = cluster_centers.size();

       for( int n = 0; n < numseeds; n++ )
       {
               int ox = cluster_centers[n].x ();//original x
               int oy = cluster_centers[n].y ();//original y
               int oind = oy*img->width + ox;

               int storeind = oind;
               for( int i = 0; i < 8; i++ )
               {
                       int nx = ox+dx8[i];//new x
                       int ny = oy+dy8[i];//new y

                       if( nx >= 0 && nx < img->width && ny >= 0 && ny < img->height)
                       {
                               int nind = ny*img->width + nx;
                               if( edges[nind] < edges[storeind])
                               {
                                       storeind = nind;
                               }
                       }
               }
               if(storeind != oind)
               {
                       cluster_centers[n] = Pixel (
                                       img->data[storeind].l (),
                                       img->data[storeind].a (),
                                       img->data[storeind].b (),
                                       storeind%img->width,
                                       storeind/img->width);
               }
       }
}

inline void
SLIC::associate_cluster_to_pixel(int vect_index, int pixel_index, int row_start, int row_length, int cluster_num)
{
	if (cluster_num >= 0 && cluster_num < state.cluster_centers.size ()
			&& cluster_num >= row_start && cluster_num < (row_start+row_length))
		state.associated_clusters_index[pixel_index*State::CLUSTER_DIRECTIONS+vect_index] = cluster_num;
	else
		state.associated_clusters_index[pixel_index*State::CLUSTER_DIRECTIONS+vect_index] = -1;
}


void
SLIC::define_image_pixels_association()
{
	const int STEP = state.region_size;
	const int sz = img->width * img->height;

	int xstrips = (0.5+float(img->width)/float(STEP));
	int ystrips = (0.5+float(img->height)/float(STEP));

	int xerr = img->width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = img->width - STEP*xstrips;}
	int yerr = img->height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = img->height- STEP*ystrips;}

	float xerrperstrip = float(xerr)/float(xstrips);
	float yerrperstrip = float(yerr)/float(ystrips);

	int numseeds = xstrips*ystrips;
	state.cluster_centers.resize (numseeds);

	int cluster_num = 0;

	for( int y = 0; y < ystrips; y++ )
	{
		int ye = y*yerrperstrip;

		for( int x = 0; x < xstrips; x++ )
		{
			int xe = x*xerrperstrip;

			// Set the cluster center.
			int seedx = (x*STEP+STEP/2+xe);
			int seedy = (y*STEP+STEP/2+ye);
			int i = seedy*img->width + seedx;

			state.cluster_centers[cluster_num] = Pixel (
					img->data[i].l (),
					img->data[i].a (),
					img->data[i].b (),
					seedx,
					seedy
					);

			// Assign pixels associativity to pixels under this cluster.
			int cluster_x_start = x*STEP + xe;
			int cluster_x_end = cluster_x_start + STEP + xerrperstrip;
			cluster_x_end = (cluster_x_end < img->width) ? cluster_x_end : img->width-1;

			int cluster_y_start = y*STEP + ye;
			int cluster_y_end = cluster_y_start + STEP + yerrperstrip;
			cluster_y_end = (cluster_y_end < img->height) ? cluster_y_end : img->height-1;

			for( int cluster_y = cluster_y_start; cluster_y <= cluster_y_end; cluster_y++)
			{
				for( int cluster_x = cluster_x_start; cluster_x <= cluster_x_end; cluster_x++)
				{
					int i = cluster_y*img->width + cluster_x;

					assert (i < sz);

					associate_cluster_to_pixel (0, i, y*xstrips, xstrips, cluster_num);
					associate_cluster_to_pixel (1, i, y*xstrips, xstrips, cluster_num-1);
					associate_cluster_to_pixel (2, i, y*xstrips, xstrips, cluster_num+1);
					associate_cluster_to_pixel (3, i, (y-1)*xstrips, xstrips, cluster_num-xstrips);
					associate_cluster_to_pixel (4, i, (y-1)*xstrips, xstrips, cluster_num-xstrips-1);
					associate_cluster_to_pixel (5, i, (y-1)*xstrips, xstrips, cluster_num-xstrips+1);
					associate_cluster_to_pixel (6, i, (y+1)*xstrips, xstrips, cluster_num+xstrips);
					associate_cluster_to_pixel (7, i, (y+1)*xstrips, xstrips, cluster_num+xstrips-1);
					associate_cluster_to_pixel (8, i, (y+1)*xstrips, xstrips, cluster_num+xstrips+1);
				}
			}


			//cout << "cluster_num=" << cluster_num << ": x=[" << cluster_x_start << ":" << cluster_x_end << "] y=["
			//												 << cluster_y_start << ":" << cluster_y_end << "]"
			//												 << "seed = [" << seedx << "," << seedy << "] " << endl;


			cluster_num++;
		}
	}

}

float
SLIC::calc_dist (const Pixel& p1, const Pixel& p2, float invwt)
{
	Pixel diff = p2 - p1;
	Pixel diff_sq = diff*diff;

	float dist_color = diff_sq.l () + diff_sq.a () + diff_sq.b ();
	float dist_xy = diff_sq.x () + diff_sq.y ();

	float dist_total = dist_color + dist_xy*invwt;
	return dist_total;
}

void
SLIC::perform_superpixel_slic_iteration ()
{
	int STEP = state.region_size;
	int sz = img->width*img->height;
	int numk = state.cluster_centers.size ();

	// ratio of how much importance to give colour over distance.
	float invwt = 1.0/((STEP/args.compactness)*(STEP/args.compactness));

	// Get reference to iteration state variables.
	vector<float>& distvec = iter_state.distvec;

	vector<Pixel> sigma (numk);
	vector<int> clustersize (numk);

	ImageRasterScan image_scan (args.access_pattern[iter_state.iter_num]);

	for (int i=0; i<sz; i++)
	{
		if (!image_scan.is_exact_index (i))
			continue;

		auto& pixel = img->data[i];


		for (int n=0; n<State::CLUSTER_DIRECTIONS; n++)
		{
			int cluster_index = state.associated_clusters_index[i*State::CLUSTER_DIRECTIONS+n];
			if (cluster_index == -1)
				continue;

			auto& current_cluster = state.cluster_centers[cluster_index];

			float dist = calc_dist (pixel, current_cluster, invwt);

			if( dist < distvec[i] )
			{
				distvec[i] = dist;
				state.labels[i] = cluster_index;
			}
		}
	}

	int ind(0);
	for( int r = 0; r < img->height; r++ )
	{
		for( int c = 0; c < img->width; c++ )
		{
			// If not fitting the pyramid scan pattern or the pixel has not
			// been assigned to a SP yet, do nothing for this pixel.
			if (image_scan.is_exact_index (ind) && state.labels[ind] != -1)
			{
				int i = c + img->width*r;
				Pixel temp (
						img->data[i].l (),
						img->data[i].a (),
						img->data[i].b (),
						c,
						r);

				assert ( state.labels[ind] >= 0 &&
							state.labels[ind] < sigma.size () &&
							state.labels[ind] < clustersize.size ());

				sigma[state.labels[ind]] = sigma[state.labels[ind]] + temp;
				clustersize[state.labels[ind]]++;
			}

			ind++;
		}
	}

	// Reset iteration_error error from previous iterations.
	iter_state.iteration_error = 0;

	for( int k = 0; k < numk; k++ )
	{
		if( clustersize[k] <= 0 ) clustersize[k] = 1;

		//computing inverse now to multiply, than divide later
		float inv = 1/float (clustersize[k]);

		Pixel new_center = sigma[k] * inv;

		// Calculate error.
		iter_state.iteration_error += new_center.get_xy_distsq_from (state.cluster_centers[k]);

		state.cluster_centers[k] = new_center;
	}

	iter_state.iteration_error = 1 - iter_state.iteration_error / state.cluster_centers.size () / (state.region_size*state.region_size*4);
	iter_state.iter_num++;

}

int
SLIC::enforce_labels_connectivity()
{
	const int K = state.cluster_centers.size ();  //the number of superpixels desired by the user
	int numlabels;	//the number of labels changes in the end if segments are removed

	int width = img->width;
	int height = img->height;
	vector<int> existing_labels = state.labels;

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
			if( 0 > state.labels[oindex] && label < state.cluster_centers.size ())
			{
				state.labels[oindex] = label;

				// Start a new segment
				xvec[0] = k;
				yvec[0] = j;

				// Quickly find an adjacent label for use later if needed
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

				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						assert (adjlabel < state.cluster_centers.size ());
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
