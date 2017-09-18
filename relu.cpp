////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#include "relu.h"


relu::relu(void)
{
	n_name = "relu";
	n_leak = 0;
}


relu::~relu(void)
{
}

void relu::load_init(ifstream &myfile, string layer_type ) {
	if (layer_type == "") {
		myfile >> layer_type;
	}
	//relu precious 0.000000 leak 0.000000
	string stemp;
	myfile >> stemp; // "n_leak"
	myfile >> n_leak;
}
void relu::save_init(ofstream &myfile) {
	myfile << endl;
	myfile << "relu leak " << n_leak << endl;
}

double relu::forward_pass() {
	n_rsp.resize(p_in1->n_rsp.size());
	//cout << "relu rsp size NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << endl;
	for (int t = 0; t < n_rsp.nchw(); t++) {
		if (p_in1->n_rsp(t) > 0) {
			n_rsp(t) = p_in1->n_rsp(t);
		}
		else {
			n_rsp(t) = p_in1->n_rsp(t)*n_leak;
		}
	}
	int stop = 1;
	return 0;
}


double relu::backward_pass() {
	n_dif.resize(n_rsp.size());
	for (int t = 0; t < n_dif.nchw() ; t++) {
		if (n_rsp(t) > 0) {
			n_dif(t) = p_out1->n_dif(t);
		}
		else {
			n_dif(t) = p_out1->n_dif(t)*n_leak;
		}
	}
	//cout << "relu partial diff size NCHW: " << n_dif.n() << " " << n_dif.c() << " " << n_dif.h() << " " << n_dif.w() << endl;
	return 0;
}