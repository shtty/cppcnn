#pragma once
#include "conv.h"
#include "relu.h"
#include "split.h"
#include "converge.h"
#include "layer_stack.h"

class residual : public layer
{
	split n_split;
	converge n_converge;
	//vector<conv> n_convs;
	conv * p_convs;
	int n_convs_size;
	vector<relu> n_relus;

	int n_n, n_c, n_h, n_w;
	float n_d;
	float n_leak;
	void delete_p_convs() {
		if ( n_convs_size > 0 ) {
			n_convs_size = 0;
			delete[] p_convs;
		}
	}
public:
	

	residual();
	~residual();
	residual(const residual &cpy);
	residual& operator=(const residual &cpy);

	int size() { return n_convs_size; }
	//void set_output(residual &out_residual);
	//void set_output(layer &out_layer);
	//void set_input(residual &in_residual);
	//void set_input(layer &in_layer);

	
	void set_size(int size = 4);
	void	set_links();
	void	n_weights_bias_set(int n, int c, int h, int w);
	void		n_weights_set(string init_method = "xavier", std::mt19937 &rng = float4d::n_random_seed);
	void set_gradient_step_size(float d);

	void set_leak(float leak = 0);

	double forward_pass();
	double backward_pass(bool update_weights = true);

	void load_init(ifstream &myfile, string layer_type = "") {
		if (layer_type == "") {
			myfile >> layer_type;
		}
		string stemp;
		int layer_number;
		myfile >> stemp; //"r_size"
		myfile >> layer_number;
		int itemp;
		myfile >> stemp; // "weight_dim"
		myfile >> itemp;
		myfile >> stemp; // "size_NCHW"

		int N, C, H, W;
		myfile >> n_n;
		myfile >> n_c;
		myfile >> n_h;
		myfile >> n_w;
		myfile >> stemp; // "leak"
		myfile >> n_leak;
		set_size(layer_number);
		n_weights_bias_set(n_n, n_c, n_h, n_w);
		set_leak(n_leak);
	}
	void save_init(ofstream &myfile) {
		myfile << endl;
		myfile << "residual r_size " << n_convs_size << " ";
		myfile << "weight_dim " << 4 << " size_NCHW " << n_n << " " << n_c << " " << n_h << " " << n_w << " ";
		myfile << "leak " << n_leak << " " << endl;
	}
	void save_weights(ofstream &myfile) {
		for (int r = 0; r < n_convs_size; r++) {
			p_convs[r].save_weights(myfile);
		}
	}
	void load_weights(ifstream &myfile) {
		for (int r = 0; r < n_convs_size; r++) {
			p_convs[r].load_weights(myfile);
		}
	}
};

