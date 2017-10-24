
#ifndef OPENC_SLIC_H_INCLUDED
#define OPENC_SLIC_H_INCLUDED

class SlicOpencl
{
	void rgb_to_lab_conversion ();

	void perform_slic_iteration_update_phase ();
	void perform_slic_iteration_accumulation_phase ();
	void perform_slic_iteration_average_phase ();

	void enforce_labels_connectivity ();
};

#endif
