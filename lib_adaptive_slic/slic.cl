// OpenCL kernel to perform an element-wise 
// add of two arrays                        
__kernel                                       
void vecadd(__global int *A,                   
            __global int *B,                   
            __global int *C)                   
{                                              
                                               
   // Get the work-item’s unique ID            
   int idx = get_global_id(0);                 
                                               
   // Add the corresponding locations of       
   // 'A' and 'B', and store the result in 'C'.
   C[idx] = A[idx] + B[idx];                   
}

void RGB2XYZ (const int sR, const int sG, const int sB, float * X, float * Y, float * Z)
{
	// sRGB (D65 illuninant assumption) to XYZ conversion
	float R = sR/255.0;
	float G = sG/255.0;
	float B = sB/255.0;

	float r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((double)(R+0.055)/1.055,(double)2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((double)(G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((double)(B+0.055)/1.055,2.4);

	*X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	*Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	*Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

void RGB2LAB (const int sR, const int sG, const int sB, float * lval, float *aval, float *bval)
{
	// sRGB to XYZ conversion
	float X, Y, Z;
	RGB2XYZ(sR, sG, sB, &X, &Y, &Z);

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
	if(xr > epsilon)	fx = pow((double)xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow((double)yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow((double)zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	*lval = 116.0*fy-16.0;
	*aval = 500.0*(fx-fy);
	*bval = 200.0*(fy-fz);
}

__kernel
void rgb2lab(__global unsigned char * img)
{
    int n = get_global_id(0);
    n *=3;

    
    int b = img[n];
            int g = img[n+1];
            int r = img[n+2];

            float l_out, a_out, b_out;
    		RGB2LAB( r, g, b, &l_out, &a_out, &b_out);

    img [n]   = (char)(l_out);
    img [n+1] = (char)(a_out);
    img [n+2] = (char)(b_out);
}


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

void getPixelFromArray(pixel * p,__global unsigned char * arr, int index)
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

}

__kernel 
void distance_update(
                     __global unsigned char * img,
                     __global int * cluster_range, 
                     __global int * misc, 
                     __global int * cluster_associativity_array, 
                     __global unsigned char * cluster_centers,
                     __global unsigned int * labels, 
                     __global unsigned int * dist_vec)
{
    int n = get_global_id(0);

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

__kernel
void distance_accumulate()
{
 int n = get_global_id(0);

}

__kernel
void center_update()
{
  int n = get_global_id(0);
}

__kernel
void update_check(__global unsigned char *  cluster_associativity_array, __global unsigned char *  talha)
{
   	// For each cluster.
    //    int n = get_global_id(0);

		// Get cluster values.
		//auto& cluster_associativity_array = cluster_associativity_array[n];

		// Check if this cluster has not converged already in last iteration.
		// All neighbourhood pixels should have movement less than a threshold.
		//for (int k=0; k<cluster_associativity_array.size (); k++)
		//{
			//int cluster_index = cluster_associativity_array[k];

			//if (cluster_index != -1)
			//{
				//if (iter_state.iteration_error_individual[cluster_index] > 0)
				//{
					//iter_state.num_clusters_updated++;
					//cluster_needs_update[n] = true;
					//break;
				//}
			//}
		//}
}
