#pragma once
#include "layer.h"
class relu : public layer
{
public:
	string layer_type() { return "relu"; }
	relu(void);
	~relu(void);
	void load_init(ifstream &myfile, string layer_type = "" );
	void save_init(ofstream &myfile);
	
	double forward_pass();
	double backward_pass();

	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
		cout << "relu leak " << leak << endl;
	}

};

