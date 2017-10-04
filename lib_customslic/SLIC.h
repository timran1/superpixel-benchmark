// SLIC.h: interface for the SLIC class.
//===========================================================================
// This code implements the superpixel method described in:
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
// "SLIC Superpixels",
// EPFL Technical Report no. 149300, June 2010.
//
// Adapted to point clouds and depth by David Stutz <david.stutz@rwth-aachen.de>
//
//===========================================================================
//	Copyright (c) 2012 Radhakrishna Achanta [EPFL]. All rights reserved.
//===========================================================================
//////////////////////////////////////////////////////////////////////

#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_

#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class CUSTOMSLIC_ARGS {
public:
	 // knobs
	 bool stateful;
     int iterations;
     int tile_square_side;
     bool one_sided_padding;
     vector<int> access_pattern;

     // outputs
	 int numlabels;	// = superpixels in main.cpp

	 // We do not mess with these
     int region_size;
     float compactness;
     bool perturbseeds;
     int color;

     CUSTOMSLIC_ARGS ()
     {
    	 tile_square_side = 0;
     }
};

class SLIC  
{
public:
	SLIC();
	virtual ~SLIC();
        //============================================================================
	// Superpixel segmentation for a given step size
	//============================================================================
        void DoSuperpixelSegmentation_ForGivenSuperpixelStep(
                const unsigned int*                             ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
                const int					width,
                const int					height,
                int*&						klabels,
				CUSTOMSLIC_ARGS & args);
	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void SaveSuperpixelLabels(
		const int*&					labels,
		const int&					width,
		const int&					height,
		const string&				filename,
		const string&				path);
	//============================================================================
	// Save supervoxel labels in a text file in raster scan, depth order
	//============================================================================
	void SaveSupervoxelLabels(
		const int**&				labels,
		const int&					width,
		const int&					height,
		const int&					depth,
		const string&				filename,
		const string&				path);
	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void DrawContoursAroundSegments(
		unsigned int*&				segmentedImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color );



	static void EnforceLabelConnectivity_extended (
			int*						klabels,//input labels that need to be corrected to remove stray labels
			const int					width,
			const int					height,
			const int&					num_sp_desired, //the number of superpixels desired by the user)
			int&						numlabels_out);//the number of labels changes in the end if segments are removed


private:
	//============================================================================
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLIC(
		vector<float>&				kseedsl,
		vector<float>&				kseedsa,
		vector<float>&				kseedsb,
		vector<float>&				kseedsx,
		vector<float>&				kseedsy,
		int*&						klabels,
		CUSTOMSLIC_ARGS & args);
        //============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetLABXYSeeds_ForGivenSeedProbabilities(
                    vector<float>&				kseedsl,
                    vector<float>&				kseedsa,
                    vector<float>&				kseedsb,
                    vector<float>&				kseedsx,
                    vector<float>&				kseedsy,
                const float&                                variance,
                const unsigned short*                        depth,
                    const int                               superpixels,
                const bool&					perturbseeds,
                const vector<float>&       edgemag);
	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetLABXYSeeds_ForGivenStepSize(
		vector<float>&				kseedsl,
		vector<float>&				kseedsa,
		vector<float>&				kseedsb,
		vector<float>&				kseedsx,
		vector<float>&				kseedsy,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<float>&		edgemag);
    //============================================================================
	// Pick seeds for supervoxels when step size of superpixels is given.
	//============================================================================
	void GetLABXYZSeeds_ForGivenStepSize(
		vector<float>&				kseedsl,
		vector<float>&				kseedsa,
		vector<float>&				kseedsb,
		vector<float>&				kseedsox,
		vector<float>&				kseedsoy,
                vector<float>&				kseedsx,
		vector<float>&				kseedsy,
                vector<float>&				kseedsz,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<float>&		edgemag);
	//============================================================================
	// Pick seeds for supervoxels
	//============================================================================
	void GetKValues_LABXYZ(
		vector<float>&				kseedsl,
		vector<float>&				kseedsa,
		vector<float>&				kseedsb,
		vector<float>&				kseedsx,
		vector<float>&				kseedsy,
		vector<float>&				kseedsz,
		const int&					STEP);
	//============================================================================
	// Move the superpixel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void PerturbSeeds(
		vector<float>&				kseedsl,
		vector<float>&				kseedsa,
		vector<float>&				kseedsb,
		vector<float>&				kseedsx,
		vector<float>&				kseedsy,
		const vector<float>&		edges);
    //============================================================================
	// Move the supervoxel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void PerturbSeeds(
		vector<float>&				kseedsl,
		vector<float>&				kseedsa,
		vector<float>&				kseedsb,
		vector<float>&				kseedsox,
		vector<float>&				kseedsoy,
                vector<float>&				kseedsx,
		vector<float>&				kseedsy,
                vector<float>&				kseedsz,
		const vector<float>&               edges);
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const float*				lvec,
		const float*				avec,
		const float*				bvec,
		const int&					width,
		const int&					height,
		vector<float>&				edges);
	//============================================================================
	// sRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		float&						X,
		float&						Y,
		float&						Z);
	//============================================================================
	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	//============================================================================
	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		float&						lval,
		float&						aval,
		float&						bval);
	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int*&		ubuff,
		float*&					lvec,
		float*&					avec,
		float*&					bvec);
	//============================================================================
	// sRGB to CIELAB conversion for 3-D volumes
	//============================================================================
	void DoRGBtoLABConversion(
		unsigned int**&				ubuff,
		float**&					lvec,
		float**&					avec,
		float**&					bvec);
	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	static void EnforceLabelConnectivity(
		const int*					labels,
		const int					width,
		const int					height,
		int*&						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user
	//============================================================================
	// Post-processing of SLIC supervoxel segmentation, to avoid stray labels.
	//============================================================================
	void EnforceSupervoxelLabelConnectivity(
		int**&						labels,//input - previous labels, output - new labels
		const int&					width,
		const int&					height,
		const int&					depth,
		int&						numlabels,
		const int&					STEP);

private:
        int							m_width;
        int							m_height;
        int							m_depth;

	float*							m_lvec;
	float*							m_avec;
	float*							m_bvec;

        float*                                                 m_xvec;
        float*                                                 m_yvec;
        float*                                                 m_zvec;
    
	float**						m_lvecvec;
	float**						m_avecvec;
	float**						m_bvecvec;
};

#endif // !defined(_SLIC_H_INCLUDED_)
