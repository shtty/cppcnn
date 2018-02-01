#pragma once

#include "layer.h"
class split: public layer
{
public:
	layer *p_out2;

	string layer_type() { return "split"; }
	split() {
		n_name = "split";
		p_out1 = NULL; p_out2 = NULL;
		p_in1 = NULL;
	}
	~split() {}
	void load_init(ifstream &myfile, string layer_type) {
		if (layer_type == "") {
			myfile >> layer_type;
		}
	}
	void save_init(ofstream &myfile) {
		myfile << endl;
		myfile << "split " << endl;
	}
	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
		cout << "split " << endl;
	}

	double forward_pass() { 
		n_rsp = p_in1->n_rsp; 
		return 0;
	}
	double backward_pass(bool update_weights = true) {
		if ( p_out1->n_dif.size() != p_out2->n_dif.size()) {
			cout << endl << "ERROR:split:backward_pass:size mismatch" << endl;
		}
		else {
			n_dif = p_out1->n_dif;
			for (int p = 0; p < n_dif.nchw(); p++ ) {
				n_dif(p) += p_out2->n_dif(p);
				n_dif(p) *= 0.5f;
			}
		}
		return 0;
	}

	
};

