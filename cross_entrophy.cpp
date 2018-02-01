////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#include "cross_entrophy.h"
//#include "cnn_kernels_000.cuh"


cross_entrophy::cross_entrophy()
{
	n_name = "cross_entrophy";
	avg_error = 0;
	all_error_for_batch = 0;
}
cross_entrophy::~cross_entrophy() {
}

double cross_entrophy::forward_pass() {
	
	
	if (p_in1->n_rsp.size() != n_rsp.size() ) {
		cout << "nchw of response and ground_truth does not match, error possibly..." << endl;
	}
	float fmax, fmin;
	fmax =  16;
	fmin = -16;
	for (int p = 0; p < n_rsp.nchw(); p++) {
		p_in1->n_rsp[p] = p_in1->n_rsp[p] < fmax ? p_in1->n_rsp[p] : fmax;
		p_in1->n_rsp[p] = p_in1->n_rsp[p] > fmin ? p_in1->n_rsp[p] : fmin;
	}
	//n_softmax.resize(n_rsp.size());
	for (int n = 0; n < n_rsp.n(); n++ ) {
		float exp_sum = 0;
		for (int p = 0; p < n_rsp.chw(); p++ ) {
			float exp_value = exp( p_in1->n_rsp(n,p) );
			//n_softmax(n, p) = exp_value;
			p_in1->n_rsp(n, p) = exp_value;
			exp_sum += exp_value;
		}
		exp_sum = exp_sum <= 0 ? 1 : exp_sum;
		for (int p = 0; p < n_rsp.chw(); p++) {
			//n_softmax(n, p) /= exp_sum;
			p_in1->n_rsp(n, p) /= exp_sum;
		}
	}

	int nchw = p_in1->n_rsp.nchw();
	all_error_for_batch = 0;
	for (int p = 0; p < nchw ; p++) {
		//all_error_for_batch -= n_rsp(p)*log(n_softmax(p));
		all_error_for_batch -= n_rsp(p)*log(p_in1->n_rsp(p));
	}
	avg_error = all_error_for_batch / double(nchw);

	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
	return avg_error;
}

double cross_entrophy::backward_pass(bool update_weights ) {
	n_dif.resize(p_in1->n_rsp.size());
	for (int p = 0; p < n_dif.nchw(); p++) {
		//n_dif[p] = n_softmax(p) - n_rsp(p);
		n_dif[p] = p_in1->n_rsp(p) - n_rsp(p);
	}
	return avg_error;
}


