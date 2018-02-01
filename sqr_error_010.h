////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "layer.h"
class sqr_error_010 : public layer
{
public:
	string layer_type() { return "sqr_error_010"; }
	
	sqr_error_010();
	~sqr_error_010();
	
	double avg_error;
	double all_error_for_batch;
	
	void save_init(ofstream &myfile) {
		myfile << endl;
		myfile << "sqr_error_010 size_NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << " " << endl;
	}

	double forward_pass();
	double backward_pass(bool update_weights = true);

	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
	}
};

