////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "layer.h"

//pool max 2 2 stride 2 pad 0 precious 0.000000
class pool : public layer
{
private:
	float4d n_backward_idx;
protected:
	
public:
	size2d  n_pool;
	size2d  n_stride;
	string pool_type;

	string layer_type() { return "pool"; }
	pool(void);
	~pool(void);
	void load_init(ifstream &myfile, string layer_type = "" );
	void save_init(ofstream &myfile);

	double forward_pass();
	double backward_pass();

	void print(bool print_n_rsp = false) {
		layer::print(print_n_rsp);
		cout << "pool " << pool_type << " " << n_pool.h << " " << n_pool.w << " stride " << n_stride.h << " " << n_stride.w << endl;
	}
};

