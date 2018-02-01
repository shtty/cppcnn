////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#include "layer.h"

layer::layer(void)
{
	n_name = "layer";
	n_use_gpu = true;
	p_in1  = NULL; 
	p_out1 = NULL;
}
layer::~layer(void)
{
}

void layer::set_input(layer &in_layer) {
	p_in1 = &in_layer;
	in_layer.p_out1 = this;
}
void layer::set_output(layer &out_layer) {
	p_out1 = &out_layer;
	out_layer.p_in1 = this;
}

void layer::load_init(ifstream &myfile, string layer_type ) {
	if (layer_type == "") {
		myfile >> layer_type;
	}
	string stemp;
	myfile >> stemp; // "size_NCHW"
	int N, C, H, W;
	myfile >> N;
	myfile >> C;
	myfile >> H;
	myfile >> W;
	n_rsp.resize(N, C, H, W);
}
void layer::save_init(ofstream &myfile) {
	myfile << "layer size_NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << " " <<endl;
}
void layer::print(bool print_values ) {
	cout << "printing layer " << n_name << endl;
	cout << "n_rsp ";
	n_rsp.print(print_values);
}
