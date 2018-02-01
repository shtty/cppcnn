#pragma once
#include "layer.h"
class converge : public layer
{
public:
	layer *p_in2;

	converge() {
		n_name = "converge";
		p_in1 = NULL; p_in2 = NULL;
		p_out1 = NULL;
	}
	~converge() {}
	void load_init(ifstream &myfile, string layer_type) {
		if (layer_type == "") {
			myfile >> layer_type;
		}
	}
	void save_init(ofstream &myfile) {
		myfile << endl;
		myfile << "converge " << endl;
	}
	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
		cout << "converge " << endl;
	}
	double forward_pass() {
		if (p_in1->n_rsp.size() != p_in2->n_rsp.size()) {
			cout << endl << "ERROR:converge:forward_pass:size mismatch" << endl;
		}
		else {
			n_rsp = p_in1->n_rsp;
			for (int p = 0; p < n_rsp.nchw(); p++) {
				n_rsp[p] += p_in2->n_rsp[p];
				n_rsp[p] *= 0.5f;
			}
		}
		return 0;
	}
	double backward_pass(bool update_weights = true) {
		n_dif = p_out1->n_dif;
		return 0;
	}
};

