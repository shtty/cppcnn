////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "layer.h"

class cross_entrophy : public layer
{
	//Implemented using following site
	//Right now cross entrophy from softmax is used 
	//http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
private:
public:
	//float4d n_softmax;
	string layer_type() { return "cross_entrophy"; }
	double avg_error;
	double all_error_for_batch;
	cross_entrophy();
	~cross_entrophy();
	void save_init(ofstream &myfile) {
		myfile << endl;
		myfile << "cross_entrophy size_NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << " " << endl;
	}

	double forward_pass();
	double backward_pass(bool update_weights = true);

	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
	}
};
