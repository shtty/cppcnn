////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "float4d.h"



class layer 
{
public:
	string n_name;
	bool   n_use_gpu;
	layer *p_in1, *p_out1;
	float4d n_rsp, n_dif; // layer response, layer chain multiplier
	
	layer(void);	
	void set_input(layer &in_layer);
	void set_output(layer &out_layer);
	
	virtual ~layer(void);
	
	virtual string layer_type() { return "layer"; }
	virtual void load_init(ifstream &myfile, string layer_type = "");
	virtual void save_init(ofstream &myfile);
	virtual void load_weights(ifstream &myfile) {}
	virtual void save_weights(ofstream &myfile) {}
	virtual void print(bool print_values = false);

	//virtual void forward_pass(layer *rsps) {}
	//virtual void backward_pass(layer *input, layer *output) {}
	//virtual void backward_pass(layer *rsps) {}
	//double forward_pass() { return 0; }
	//double backward_pass(bool update_weights = true) { return 0; }
	virtual double forward_pass() { return 0; }
	virtual double backward_pass(bool update_weights = true) { return 0; }
	
	virtual void set_gradient_step_size(float d) {}
	virtual void n_weights_bias_set(int n, int c, int h, int w) {}
	virtual void n_weights_set(string init_method, std::mt19937 &rng = float4d::n_random_seed) {}
	
	
};
