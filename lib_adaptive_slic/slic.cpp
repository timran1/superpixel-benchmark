#include <cmath>
#include <iostream>
#include <assert.h>

#include "slic.h"
#include "adaptive_slic.h"
// OpenCL includes
#include <CL/cl.h>

using namespace std;


typedef struct pixel_Struct
  {
    char color[3];
    unsigned char pad[1];
    int           coord[2];
  } pixel;

unsigned int calc_dist(pixel* p1, pixel * p2, int invwt) {
        int temp[5];

        for (int i =0; i < 3; i ++)
        {
            temp[i] = p1->color[i] - p2->color[i];
            temp[i] *= temp[i];
        }

        for (int i =0; i < 2; i ++)
        {
            temp[i+3] =p1->coord[i] - p2->coord[i];
            temp[i+3] *= temp[i+3];
        }


	float dist_color = temp[0] + temp[1] + temp[2];
	float dist_xy = temp[3] + temp[4];

	float dist_total = dist_color + dist_xy * invwt;
	dist_total = sqrt(dist_total);
	return (unsigned int)(dist_total);
}

void getPixelFromArray(pixel * p,unsigned char * arr, int index)
{
        for (int i = 0; i < 3; i++)
        {
            p->color[i] = arr[index++];
        }
		index++;
        for (int i = 0; i < 2; i++)
        {
            p->coord[i] = arr[index] | ((int)arr[index+1] << 8) | ((int)arr[index+2] << 16) | ((int)arr[index+3] << 24);
            index+=4;
        }

        char val0 = p->color[0];
        char val1 = p->color[1];
        char val2 = p->color[2];
        int val3 = p->coord[0];
        int val4 = p->coord[1];

}

void distance_update(int n,
                     unsigned char * img,
                     int * cluster_range,
                     int * misc,
                     int * cluster_associativity_array,
                     unsigned char * cluster_centers,
                     int * labels,
                     int * dist_vec)
{

    int sPixel = sizeof(pixel);
	int width = misc[0];
	int invwt = misc[1];
    int range_iter = n*4;
    int ass_iter = n*9;

    // For each cluster.
  //		if (!cluster_needs_update[n])
//			continue;

		// Get cluster values.
		int x1 = cluster_range[range_iter];
                int x2 = cluster_range[range_iter+1];
                int y1 = cluster_range[range_iter+2];
                int y2 = cluster_range[range_iter+3];

		// Update distance for all pixels in current cluster.
		for (int x = x1; x < x2; x++) {
			for (int y = y1; y < y2; y++) {

				// For each pixel in cluster.
				int i = y * width + x;
                int itemp = i *sPixel;
while (i == 72);
			//	if (!image_scan.is_exact_index(i))
			//		continue;

				pixel p;
                getPixelFromArray (&p, img, itemp);

				for (int k = 0; k < 9; k++) {
					int cluster_index = cluster_associativity_array[k + ass_iter];

					if (cluster_index >= 0) {
						pixel current_cluster;
						getPixelFromArray(&current_cluster ,cluster_centers, (cluster_index * sPixel));

						unsigned int  dist = calc_dist(&p, &current_cluster, invwt);

						if (dist < dist_vec[i]) {
							dist_vec[i] = dist;
							labels[i] = cluster_index;

						}

					}

				}
			}
		}
}

//-----------------------------------------------------------------------
// SLIC class.
//-----------------------------------------------------------------------
SLIC::SLIC(shared_ptr<Image> img, AdaptiveSlicArgs& args) :
		img(img), args(args) {
	state.init(args, img->width * img->height);
}

void SLIC::init_iteration_state() {
	iter_state.init(img->width * img->height, state.cluster_centers.size());
}

void SLIC::get_labels_mat(cv::Mat &labels) {
	// Convert labels.
	for (int i = 0; i < img->height; ++i) {
		for (int j = 0; j < img->width; ++j) {
			labels.at<int>(i, j) = state.labels[j + i * img->width];
		}
	}
}

void SLIC::set_initial_seeds(cv::Mat grid_mat) {
	define_image_pixels_association(grid_mat);

	if (args.perturbseeds) {
		vector<float> edges;
		detect_lab_edges(edges);
		perturb_seeds(edges);
	}

	// compute region size for number of clusters actually created. args.region_size is
	// used to determine search area during SLIC iterations.
	state.update_region_size_from_sp(img->width * img->height,
			state.cluster_centers.size());
}

void SLIC::detect_lab_edges(vector<float>& edges) {
	int& width = img->width;
	int& height = img->height;

	int sz = width * height;

	edges.resize(sz, 0);
	for (int j = 1; j < img->height - 1; j++) {
		for (int k = 1; k < img->width - 1; k++) {
			int i = j * img->width + k;
			float dx = (img->data[i - 1].l() - img->data[i + 1].l())
					* (img->data[i - 1].l() - img->data[i + 1].l())
					+ (img->data[i - 1].a() - img->data[i + 1].a())
							* (img->data[i - 1].a() - img->data[i + 1].a())
					+ (img->data[i - 1].b() - img->data[i + 1].b())
							* (img->data[i - 1].b() - img->data[i + 1].b());

			float dy = (img->data[i - width].l() - img->data[i + width].l())
					* (img->data[i - width].l() - img->data[i + width].l())
					+ (img->data[i - width].a() - img->data[i + width].a())
							* (img->data[i - width].a()
									- img->data[i + width].a())
					+ (img->data[i - width].b() - img->data[i + width].b())
							* (img->data[i - width].b()
									- img->data[i + width].b());

			edges[i] = dx * dx + dy * dy;
		}
	}
}

void SLIC::perturb_seeds(const vector<float>& edges) {
	vector < Pixel > &cluster_centers = state.cluster_centers;

	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int numseeds = cluster_centers.size();

	for (int n = 0; n < numseeds; n++) {
		int ox = cluster_centers[n].x();    //original x
		int oy = cluster_centers[n].y();    //original y
		int oind = oy * img->width + ox;

		int storeind = oind;
		for (int i = 0; i < 8; i++) {
			int nx = ox + dx8[i];    //new x
			int ny = oy + dy8[i];    //new y

			if (nx >= 0 && nx < img->width && ny >= 0 && ny < img->height) {
				int nind = ny * img->width + nx;
				if (edges[nind] < edges[storeind]) {
					storeind = nind;
				}
			}
		}
		if (storeind != oind) {
			cluster_centers[n] = Pixel(img->data[storeind].l(),
					img->data[storeind].a(), img->data[storeind].b(),
					storeind % img->width, storeind / img->width);
		}
	}
}

inline void SLIC::associate_cluster_to_pixel(int vect_index, int cluster_index,
		int row_start, int row_length, int cluster_num) {
	int index = cluster_index * 9 + vect_index;
	assert(state.cluster_associativity_array[index] == -1);

	if (cluster_num >= 0 && cluster_num < state.cluster_centers.size()
			&& cluster_num >= row_start
			&& cluster_num < (row_start + row_length))
		state.cluster_associativity_array[index] = cluster_num;
	else
		state.cluster_associativity_array[index] = -1;
}

void SLIC::define_image_pixels_association(cv::Mat grid_mat) {
	const int STEP = state.region_size;
	const int sz = img->width * img->height;

	int xstrips = (0.5 + float(img->width) / float(STEP));
	int ystrips = (0.5 + float(img->height) / float(STEP));

	int xerr = img->width - STEP * xstrips;
	if (xerr < 0) {
		xstrips--;
		xerr = img->width - STEP * xstrips;
	}
	int yerr = img->height - STEP * ystrips;
	if (yerr < 0) {
		ystrips--;
		yerr = img->height - STEP * ystrips;
	}

	float xerrperstrip = float(xerr) / float(xstrips);
	float yerrperstrip = float(yerr) / float(ystrips);

	int numseeds = xstrips * ystrips;
	state.cluster_centers.resize(numseeds);
	state.cluster_range.resize(numseeds * 4);
	state.cluster_associativity_array.assign(numseeds * 9, -1);

	int cluster_num = 0;

	int last_cluster_end_y = 0;
	for (int y = 0; y < ystrips; y++) {
		int ye = y * yerrperstrip;

		int cluster_y_start = last_cluster_end_y;
		int cluster_y_end = (y + 1) * (STEP + yerrperstrip);
		cluster_y_end =
				(cluster_y_end < (img->height - 2)) ?
						cluster_y_end : img->height;

		last_cluster_end_y = cluster_y_end;

		// Draw this cluster boundary on grid_map
		cv::line(grid_mat, cv::Point(0, cluster_y_end + 1),
				cv::Point(img->width, cluster_y_end + 1), cv::Scalar(255));

		int last_cluster_end_x = 0;
		for (int x = 0; x < xstrips; x++) {
			int xe = x * xerrperstrip;

			// Set the cluster center.
			int seedx = (x * STEP + STEP / 2 + xe);
			int seedy = (y * STEP + STEP / 2 + ye);
			int i = seedy * img->width + seedx;

			state.cluster_centers[cluster_num] = Pixel(img->data[i].l(),
					img->data[i].a(), img->data[i].b(), seedx, seedy);

			// Assign pixels associativity to pixels under this cluster.
			int cluster_x_start = last_cluster_end_x;
			int cluster_x_end = (x + 1) * (STEP + xerrperstrip);
			cluster_x_end =
					(cluster_x_end < (img->width - 2)) ?
							cluster_x_end : img->width;

			last_cluster_end_x = cluster_x_end;

			state.cluster_range[cluster_num * 4 + 0] = cluster_x_start;
			state.cluster_range[cluster_num * 4 + 1] = cluster_x_end;
			state.cluster_range[cluster_num * 4 + 2] = cluster_y_start;
			state.cluster_range[cluster_num * 4 + 3] = cluster_y_end;

			associate_cluster_to_pixel(0, cluster_num, y * xstrips, xstrips,
					cluster_num);
			associate_cluster_to_pixel(1, cluster_num, y * xstrips, xstrips,
					cluster_num - 1);
			associate_cluster_to_pixel(2, cluster_num, y * xstrips, xstrips,
					cluster_num + 1);
			associate_cluster_to_pixel(3, cluster_num, (y - 1) * xstrips,
					xstrips, cluster_num - xstrips);
			associate_cluster_to_pixel(4, cluster_num, (y - 1) * xstrips,
					xstrips, cluster_num - xstrips - 1);
			associate_cluster_to_pixel(5, cluster_num, (y - 1) * xstrips,
					xstrips, cluster_num - xstrips + 1);
			associate_cluster_to_pixel(6, cluster_num, (y + 1) * xstrips,
					xstrips, cluster_num + xstrips);
			associate_cluster_to_pixel(7, cluster_num, (y + 1) * xstrips,
					xstrips, cluster_num + xstrips - 1);
			associate_cluster_to_pixel(8, cluster_num, (y + 1) * xstrips,
					xstrips, cluster_num + xstrips + 1);

			// Draw this cluster boundary on grid_map. Only need to do this on first row.
			if (y == 0)
				cv::line(grid_mat, cv::Point(cluster_x_start, 0),
						cv::Point(cluster_x_start, img->height),
						cv::Scalar(255));

			//cout << "cluster_num=" << cluster_num << ": x=[" << cluster_x_start << ":" << cluster_x_end << "] y=["
			//												 << cluster_y_start << ":" << cluster_y_end << "]"
			//												 << "seed = [" << seedx << "," << seedy << "] " << endl;

			cluster_num++;
		}
	}

}

word SLIC::calc_dist(const Pixel& p1, const Pixel& p2, float invwt) {
	Pixel diff = p2 - p1;
	vector<int> diff_sq = diff.get_int_arr();
	for (int i = 0; i < 5; i++)
		diff_sq[i] *= diff_sq[i];

	float dist_color = diff_sq[0] + diff_sq[1] + diff_sq[2];
	float dist_xy = diff_sq[3] + diff_sq[4];

	float dist_total = dist_color + dist_xy * invwt;
	dist_total = sqrt(dist_total);
	return word(dist_total);
}

void SLIC::perform_superpixel_slic_iteration() {
	Profiler profiler("perform_superpixel_slic_iteration", false);

	int STEP = state.region_size;
	int sz = img->width * img->height;
	int numk = state.cluster_centers.size();

	// ratio of how much importance to give colour over distance.
	char invwt = 1.0 / (STEP / args.compactness);
	float error_normalization = state.region_size * state.region_size * 4;

	// Reset iteration_variables from previous iterations.
	iter_state.iteration_error = 0;
	iter_state.num_clusters_updated = 0;

	vector<vector<int>> sigma(numk, vector<int>(5));
	vector < word > clustersize(numk);
	vector<bool> cluster_needs_update(numk, false);

	ImageRasterScan image_scan(args.access_pattern[iter_state.iter_num]);

	// For each cluster.
	for (int n = 0; n < state.cluster_centers.size(); n++) {

		// Check if this cluster has not converged already in last iteration.
		// All neighbourhood pixels should have movement less than a threshold.
		for (int k = 0; k < 9; k++) {
			assert((n * 9 + k) < state.cluster_associativity_array.size());
			int cluster_index = state.cluster_associativity_array[n * 9 + k];

			if (cluster_index != -1) {
				if (iter_state.iteration_error_individual[cluster_index] > 0) {
					iter_state.num_clusters_updated++;
					cluster_needs_update[n] = true;
					break;
				}
			}
		}
	}
	profiler.update_checkpoint("1st pass");


	int misc[] = {img->width, invwt};
	vector<int> labels_temp (sz, -1);
	vector<int> distvec_temp (sz, INT_MAX);

	for (int mm = 0; mm <sz; mm++)
		labels_temp[mm] = state.labels[mm];

	for (int mm = 0; mm <sz; mm++)
		distvec_temp[mm] = iter_state.distvec[mm];

	for (int n = 0; n < state.cluster_centers.size(); n++)
	{
       distance_update(n, (unsigned char *)&img->data[0], &state.cluster_range[0], misc, &state.cluster_associativity_array[0], (unsigned char *) &state.cluster_centers[0],&labels_temp[0],&distvec_temp[0]);
	}

	// For each cluster.
	for (int n = 0; n < state.cluster_centers.size(); n++)
	//tbb::parallel_for( tbb::blocked_range<size_t>(0,numk), [=](const tbb::blocked_range<size_t>& r) {
	//		for(size_t n=r.begin(); n!=r.end(); ++n)
			{
		if (!cluster_needs_update[n])
			continue;
		
		assert((n * 4 + 3) < state.cluster_range.size());
		int x1 = state.cluster_range[n * 4 + 0];
		int x2 = state.cluster_range[n * 4 + 1];
		int y1 = state.cluster_range[n * 4 + 2];
		int y2 = state.cluster_range[n * 4 + 3];

		// Update distance for all pixels in current cluster.
		for (int x = x1; x < x2; x++) {
			for (int y = y1; y < y2; y++) {
				// For each pixel in cluster.
				int i = y * img->width + x;

				while(i==72);
				if (!image_scan.is_exact_index(i))
					continue;

				auto& pixel = img->data[i];

				for (int k = 0; k < 9; k++) {
					int cluster_index = state.cluster_associativity_array[n * 9
							+ k];

					if (cluster_index != -1) {
						auto& current_cluster =
								state.cluster_centers[cluster_index];

						word dist = calc_dist(pixel, current_cluster, invwt);

						if (dist < iter_state.distvec[i]) {
							iter_state.distvec[i] = dist;
							state.labels[i] = cluster_index;
						}
					}

				}
			}
		}
	}
	//});
	profiler.update_checkpoint("2nd pass");


	for (int mm = 0; mm <sz; mm++)
		if (labels_temp[mm] != state.labels[mm])
			cout << "labels_temp Error: " << mm << endl;

	for (int mm = 0; mm <sz; mm++)
		if (distvec_temp[mm] != iter_state.distvec[mm])
			cout << "distvec_temp Error: " << mm << endl;



	// For each cluster.
	for (int n = 0; n < state.cluster_centers.size(); n++) {
		if (!cluster_needs_update[n])
			continue;

		assert((n * 4 + 3) < state.cluster_range.size());
		int x1 = state.cluster_range[n * 4 + 0];
		int x2 = state.cluster_range[n * 4 + 1];
		int y1 = state.cluster_range[n * 4 + 2];
		int y2 = state.cluster_range[n * 4 + 3];

		// Update distance for all pixels in current cluster.
		for (int x = x1; x < x2; x++) {
			for (int y = y1; y < y2; y++) {
				// For each pixel in cluster.
				int i = y * img->width + x;

				// If not fitting the pyramid scan pattern or the pixel has not
				// been assigned to a SP yet, do nothing for this pixel.
				if (image_scan.is_exact_index(i) && state.labels[i] != -1) {
					assert(
							state.labels[i] >= 0
									&& state.labels[i] < sigma.size()
									&& state.labels[i] < clustersize.size());

					sigma[state.labels[i]][0] += img->data[i].l();
					sigma[state.labels[i]][1] += img->data[i].a();
					sigma[state.labels[i]][2] += img->data[i].b();
					sigma[state.labels[i]][3] += img->data[i].x();
					sigma[state.labels[i]][4] += img->data[i].y();

					clustersize[state.labels[i]]++;
				}

			}
		}
	}
	profiler.update_checkpoint("3rd pass");

	// For each cluster.
	for (int n = 0; n < numk; n++) {
		if (!cluster_needs_update[n])
			continue;

		if (clustersize[n] <= 0)
			clustersize[n] = 1;

		vector<int> new_center_int(5);
		for (int i = 0; i < 5; i++)
			new_center_int[i] = sigma[n][i] / clustersize[n];

		Pixel new_center(new_center_int);

		// Calculate error.
		Pixel diff = new_center - state.cluster_centers[n];

		byte current_cluster_error = diff.get_mag();// new_center.get_xy_distsq_from (state.cluster_centers[n]);

		iter_state.iteration_error_individual[n] = current_cluster_error;
		iter_state.iteration_error += current_cluster_error;

		// Update cluster center.
		state.cluster_centers[n] = new_center;
	}
	profiler.update_checkpoint("4th pass");

	iter_state.iteration_error = iter_state.iteration_error
			/ state.cluster_centers.size() / error_normalization;
	iter_state.iter_num++;

	std::cout << profiler.print_checkpoints();
}

int SLIC::enforce_labels_connectivity() {
	const int K = state.cluster_centers.size(); //the number of superpixels desired by the user
	int numlabels; //the number of labels changes in the end if segments are removed

	int width = img->width;
	int height = img->height;
	vector<int> existing_labels = state.labels;

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	const int sz = width * height;
	const int SUPSZ = sz / K;
	state.labels.assign(sz, -1);
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);	//adjacent label
	for (int j = 0; j < height; j++) {
		for (int k = 0; k < width; k++) {
			if (0 > state.labels[oindex]
					&& label < state.cluster_centers.size()) {
				state.labels[oindex] = label;

				// Start a new segment
				xvec[0] = k;
				yvec[0] = j;

				// Quickly find an adjacent label for use later if needed
				{
					for (int n = 0; n < 4; n++) {
						int x = xvec[0] + dx4[n];
						int y = yvec[0] + dy4[n];
						if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
							int nindex = y * width + x;
							if (state.labels[nindex] >= 0)
								adjlabel = state.labels[nindex];
						}
					}
				}

				int count(1);
				for (int c = 0; c < count; c++) {
					for (int n = 0; n < 4; n++) {
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
							int nindex = y * width + x;

							if (0 > state.labels[nindex]
									&& existing_labels[oindex]
											== existing_labels[nindex]) {
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
				if (count <= SUPSZ >> 2) {
					for (int c = 0; c < count; c++) {
						int ind = yvec[c] * width + xvec[c];
						assert(adjlabel < state.cluster_centers.size());
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

	if (xvec)
		delete[] xvec;
	if (yvec)
		delete[] yvec;

	return numlabels;
}

void SLIC::RGB2XYZ(const int sR, const int sG, const int sB, float& X, float& Y,
		float& Z) {
	// sRGB (D65 illuninant assumption) to XYZ conversion
	float R = sR / 255.0;
	float G = sG / 255.0;
	float B = sB / 255.0;

	float r, g, b;

	if (R <= 0.04045)
		r = R / 12.92;
	else
		r = pow((R + 0.055) / 1.055, 2.4);
	if (G <= 0.04045)
		g = G / 12.92;
	else
		g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)
		b = B / 12.92;
	else
		b = pow((B + 0.055) / 1.055, 2.4);

	X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
	Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
	Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

void SLIC::RGB2LAB(const int sR, const int sG, const int sB, float& lval,
		float& aval, float& bval) {
	// sRGB to XYZ conversion
	float X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	// XYZ to LAB conversion
	float epsilon = 0.008856;	//actual CIE standard
	float kappa = 903.3;		//actual CIE standard

	float Xr = 0.950456;	//reference white
	float Yr = 1.0;		//reference white
	float Zr = 1.088754;	//reference white

	float xr = X / Xr;
	float yr = Y / Yr;
	float zr = Z / Zr;

	float fx, fy, fz;
	if (xr > epsilon)
		fx = pow(xr, 1.0 / 3.0);
	else
		fx = (kappa * xr + 16.0) / 116.0;
	if (yr > epsilon)
		fy = pow(yr, 1.0 / 3.0);
	else
		fy = (kappa * yr + 16.0) / 116.0;
	if (zr > epsilon)
		fz = pow(zr, 1.0 / 3.0);
	else
		fz = (kappa * zr + 16.0) / 116.0;

	lval = 116.0 * fy - 16.0;
	aval = 500.0 * (fx - fy);
	bval = 200.0 * (fy - fz);
}

void SLIC::do_rgb_to_lab_conversion(const cv::Mat &mat, cv::Mat &out,
		int padding_c_left, int padding_r_up) {
	assert(
			mat.rows + padding_r_up <= out.rows
					&& mat.cols + padding_c_left <= out.cols);

	// Ranges:
	// L in [0, 100]
	// A in [-86.185, 98,254]
	// B in [-107.863, 94.482]

	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {

			int b = mat.at < cv::Vec3b > (i, j)[0];
			int g = mat.at < cv::Vec3b > (i, j)[1];
			int r = mat.at < cv::Vec3b > (i, j)[2];

			int arr_index = j + mat.cols * i;

			float l_out, a_out, b_out;
			RGB2LAB(r, g, b, l_out, a_out, b_out);

			out.at < cv::Vec3b > (i + padding_r_up, j + padding_c_left)[0] =
					char(l_out);
			out.at < cv::Vec3b > (i + padding_r_up, j + padding_c_left)[1] =
					char(a_out);
			out.at < cv::Vec3b > (i + padding_r_up, j + padding_c_left)[2] =
					char(b_out);

		}
	}
}


void hwSLIC::do_rgb_to_lab_conversion(const cv::Mat &mat, cv::Mat &out,
		int padding_c_left, int padding_r_up, openCL& hw) {

	//std::vector<uchar> array(mat.rows*mat.cols);
	assert(mat.isContinuous());

	hw.buffers["img_lab"] = std::make_shared < cl::Buffer
			> (*(hw.context.get()), CL_MEM_READ_WRITE, mat.rows * mat.cols * 3);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["img_lab"].get()), CL_FALSE, 0,
			mat.rows * mat.cols * 3, mat.data);

	//rgb2lab((unsigned char*) mat.data);

	cl::NDRange globalws(mat.rows * mat.cols);

	hw.kernels["rgb2lab"]->setArg(0, *hw.buffers["img_lab"].get());

	int status = hw.queues[0].enqueueNDRangeKernel(*hw.kernels["rgb2lab"].get(),
			cl::NullRange, globalws, cl::NullRange);
	if (!status)
		cout << status << endl;

	status = hw.queues[0].finish();

	hw.oclCheckError("Queue finish", status);

	out.create(mat.rows, mat.cols, mat.type());

	// Read the output buffer back to the host
	status = hw.queues[0].enqueueReadBuffer(*hw.buffers["img_lab"].get(),
			CL_TRUE, 0, out.rows * out.cols * 3, out.data);

	hw.oclCheckError("Read finish", status);

}

void hwSLIC::perform_superpixel_slic_iteration(openCL& hw) {
	Profiler profiler("perform_superpixel_slic_iteration", true);

	int STEP = state.region_size;
	int sz = img->width * img->height;
	int numk = state.cluster_centers.size();

	// ratio of how much importance to give colour over distance.
	char invwt = 1.0 / (STEP / args.compactness);
	float error_normalization = state.region_size * state.region_size * 4;

	// Reset iteration_variables from previous iterations.
	iter_state.iteration_error = 0;
	iter_state.num_clusters_updated = 0;

	vector<vector<int>> sigma(numk, vector<int>(5));
	vector < word > clustersize(numk);
	vector<bool> cluster_needs_update(numk, false);

	ImageRasterScan image_scan(args.access_pattern[iter_state.iter_num]);

	// For each cluster.
	for (int n = 0; n < state.cluster_centers.size(); n++) {
		// Check if this cluster has not converged already in last iteration.
		// All neighbourhood pixels should have movement less than a threshold.
		for (int k = 0; k < 9; k++) {
			int cluster_index = state.cluster_associativity_array[n * 9 + k];

			if (cluster_index != -1) {
				if (iter_state.iteration_error_individual[cluster_index] > 0) {
					iter_state.num_clusters_updated++;
					cluster_needs_update[n] = true;
					break;
				}
			}
		}
	}
	profiler.update_checkpoint("1st pass");

	// For each cluster.
//
	int misc[] = {img->width, invwt};
//	if (false)
//	{
//		for (int n = 0; n < state.cluster_centers.size(); n++)
//		{
//	       distance_update(n, (unsigned char *)&img->data[0], &state.cluster_range[0], misc, &state.cluster_associativity_array[0], (unsigned char *) &state.cluster_centers[0],&state.labels[0],&iter_state.distvec[0]);
//		}
//	}
//	else
//	{
	
	/* NDRange*/
	hw.buffers["lab_big"] = std::make_shared < cl::Buffer> (*(hw.context.get()), CL_MEM_READ_WRITE, img->data.size() * sizeof(Pixel));
	hw.buffers["cluster_range"] =
			std::make_shared < cl::Buffer
					> (*(hw.context.get()), CL_MEM_READ_WRITE, state.cluster_range.size()
							* 4);
	hw.buffers["misc"] = std::make_shared < cl::Buffer
			> (*(hw.context.get()), CL_MEM_READ_WRITE, sizeof (misc));
	hw.buffers["cluster_associativity_array"] =
			std::make_shared < cl::Buffer
					> (*(hw.context.get()), CL_MEM_READ_WRITE, state.cluster_associativity_array.size()
							* 4);
	hw.buffers["cluster_centers"] =
			std::make_shared < cl::Buffer
					> (*(hw.context.get()), CL_MEM_READ_WRITE, state.cluster_centers.size()
							* sizeof(Pixel));
	hw.buffers["labels"] = std::make_shared < cl::Buffer
			> (*(hw.context.get()), CL_MEM_READ_WRITE, state.labels.size() * 4);
	hw.buffers["dist_vec"] = std::make_shared < cl::Buffer
			> (*(hw.context.get()), CL_MEM_READ_WRITE, sz * sizeof(int));
	

	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["lab_big"].get()), CL_TRUE, 0, img->data.size() * sizeof(Pixel), &img->data[0]);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["cluster_range"].get()), CL_TRUE, 0, state.cluster_range.size() * 4, &state.cluster_range[0]);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["misc"].get()), CL_TRUE, 0, sizeof (misc), misc);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["cluster_associativity_array"].get()), CL_TRUE, 0, state.cluster_associativity_array.size() * 4, &state.cluster_associativity_array[0]);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["cluster_centers"].get()), CL_TRUE, 0, state.cluster_centers.size() * sizeof(Pixel), &state.cluster_centers[0]);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["labels"].get()), CL_TRUE, 0, state.labels.size() * 4, &state.labels[0]);
	hw.queues[0].enqueueWriteBuffer(*(hw.buffers["dist_vec"].get()), CL_TRUE, 0, sz * sizeof(int), &iter_state.distvec[0]);
	
	

	//rgb2lab((unsigned char*) mat.data);

	cl::NDRange globalws(numk);

	hw.kernels["distance_update"]->setArg(0, *hw.buffers["lab_big"].get());
	hw.kernels["distance_update"]->setArg(1, *hw.buffers["cluster_range"].get());
	hw.kernels["distance_update"]->setArg(2, *hw.buffers["misc"].get());
	hw.kernels["distance_update"]->setArg(3, *hw.buffers["cluster_associativity_array"].get());
	hw.kernels["distance_update"]->setArg(4, *hw.buffers["cluster_centers"].get());
	hw.kernels["distance_update"]->setArg(5, *hw.buffers["labels"].get());
	hw.kernels["distance_update"]->setArg(6, *hw.buffers["dist_vec"].get());


	int status = hw.queues[0].enqueueNDRangeKernel(*hw.kernels["distance_update"].get(),
			cl::NullRange, globalws, cl::NullRange);
	if (!status)
		cout << status << endl;

	status = hw.queues[0].finish();

	hw.oclCheckError("Queue finish", status);
	
//	status = hw.queues[0].flush();

	// Read the output buffer back to the host
	status = hw.queues[0].enqueueReadBuffer(*(hw.buffers["labels"].get()), CL_TRUE, 0, state.labels.size() * 4, &state.labels[0]);
    status = hw.queues[0].enqueueReadBuffer(*(hw.buffers["dist_vec"].get()), CL_TRUE, 0, sz * sizeof(int), &iter_state.distvec[0]);
   
    


    hw.oclCheckError("Read", status);

	profiler.update_checkpoint("2nd pass");

	// For each cluster.
	for (int n = 0; n < state.cluster_centers.size(); n++) {
		if (!cluster_needs_update[n])
			continue;

		int x1 = state.cluster_range[n * 4 + 0];
		int x2 = state.cluster_range[n * 4 + 1];
		int y1 = state.cluster_range[n * 4 + 2];
		int y2 = state.cluster_range[n * 4 + 3];

		// Update distance for all pixels in current cluster.
		for (int x = x1; x < x2; x++) {
			for (int y = y1; y < y2; y++) {
				// For each pixel in cluster.
				int i = y * img->width + x;

				// If not fitting the pyramid scan pattern or the pixel has not
				// been assigned to a SP yet, do nothing for this pixel.
				if (image_scan.is_exact_index(i) && state.labels[i] != -1) {
					assert(
							state.labels[i] >= 0
									&& state.labels[i] < sigma.size()
									&& state.labels[i] < clustersize.size());

					sigma[state.labels[i]][0] += img->data[i].l();
					sigma[state.labels[i]][1] += img->data[i].a();
					sigma[state.labels[i]][2] += img->data[i].b();
					sigma[state.labels[i]][3] += img->data[i].x();
					sigma[state.labels[i]][4] += img->data[i].y();

					clustersize[state.labels[i]]++;
				}

			}
		}
	}
	profiler.update_checkpoint("3rd pass");

	// For each cluster.
	for (int n = 0; n < numk; n++) {
		if (!cluster_needs_update[n])
			continue;

		if (clustersize[n] <= 0)
			clustersize[n] = 1;

		vector<int> new_center_int(5);
		for (int i = 0; i < 5; i++)
			new_center_int[i] = sigma[n][i] / clustersize[n];

		Pixel new_center(new_center_int);

		// Calculate error.
		Pixel diff = new_center - state.cluster_centers[n];

		byte current_cluster_error = diff.get_mag();// new_center.get_xy_distsq_from (state.cluster_centers[n]);

		iter_state.iteration_error_individual[n] = current_cluster_error;
		iter_state.iteration_error += current_cluster_error;

		// Update cluster center.
		state.cluster_centers[n] = new_center;
	}
	profiler.update_checkpoint("4th pass");

	iter_state.iteration_error = iter_state.iteration_error
			/ state.cluster_centers.size() / error_normalization;
	iter_state.iter_num++;

	std::cout << profiler.print_checkpoints();
}
