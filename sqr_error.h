#pragma once
#include "layer.h"

class sqr_error : public layer
{
public:
	string layer_type() { return "sqr_error"; }
	double avg_error;
	double all_error_for_batch;
	sqr_error();
	~sqr_error();
	void save_init(ofstream &myfile) {
		myfile << endl;
		myfile << "sqr_error size_NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << endl;
	}

	double forward_pass();
	double backward_pass();

	void backward_pass_mask( layer *rsps, float mask_value );
	void backward_pass_mask(float mask_value);
	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
	}
};
