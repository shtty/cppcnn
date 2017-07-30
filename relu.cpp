#include "relu.h"


relu::relu(void)
{
	n_name = "relu";
	leak = 0;
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
	myfile >> stemp; // "leak"
	myfile >> leak;
}
void relu::save_init(ofstream &myfile) {
	myfile << endl;
	myfile << "relu leak " << leak << endl;
}

double relu::forward_pass() {
	n_rsp.resize(n_in1->n_rsp.size());
	//cout << "relu rsp size NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << endl;
	for (int t = 0; t < n_rsp.nchw(); t++) {
		if (n_in1->n_rsp(t) > 0) {
			n_rsp(t) = n_in1->n_rsp(t);
		}
		else {
			n_rsp(t) = n_in1->n_rsp(t)*leak;
		}
	}
	int stop = 1;
	return 0;
}


double relu::backward_pass() {
	n_dif.resize(n_rsp.size());
	for (int t = 0; t < n_dif.nchw() ; t++) {
		if (n_rsp(t) > 0) {
			n_dif(t) = n_out1->n_dif(t);
		}
		else {
			n_dif(t) = n_out1->n_dif(t)*leak;
		}
	}
	//cout << "relu partial diff size NCHW: " << n_dif.n() << " " << n_dif.c() << " " << n_dif.h() << " " << n_dif.w() << endl;
	return 0;
}