////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#include "sqr_error.h"
//#include "cnn_kernels_000.cuh"


sqr_error::sqr_error()
{
	n_name = "sqr_error";
	avg_error = 0;
	all_error_for_batch = 0;
}
sqr_error::~sqr_error() {
}

double sqr_error::forward_pass() {
	all_error_for_batch = 0;
	int nchw = p_in1->n_rsp.nchw();
	if (nchw != n_rsp.nchw()) {
		cout << "nchw of response and sqr_error_011 does not match, using smaller nchw..." << endl;
		if (nchw > n_rsp.nchw()) {
			nchw = n_rsp.nchw();
		}
	}
	for (int p = 0; p < nchw ; p++) {
		all_error_for_batch += double(p_in1->n_rsp(p) - n_rsp(p))*double(p_in1->n_rsp(p) - n_rsp(p));
	}
	avg_error = all_error_for_batch / double(nchw);
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
	return avg_error;
}

double sqr_error::backward_pass() {
	n_dif.resize(p_in1->n_rsp.size());
	float inv_psize = 1 / float(n_dif.nchw());
	for (int p = 0; p < n_dif.nchw(); p++) {
		n_dif(p) = 2 * inv_psize*(p_in1->n_rsp(p) - n_rsp(p));
	}
	return avg_error;
}

void sqr_error::backward_pass_mask(layer *rsps, float mask_value) {
	n_dif.resize(rsps->n_rsp.size());
	all_error_for_batch = 0;
	float pcount = 0;
	for (int p = 0; p < n_dif.nchw(); p++) {
		if ( n_rsp(p) != mask_value) {
			all_error_for_batch += double(rsps->n_rsp(p) - n_rsp(p))*double(rsps->n_rsp(p) - n_rsp(p));
			pcount++;
		}
	}
	avg_error = all_error_for_batch / double(pcount);
	float inv_psize = 1 / pcount;
	for (int p = 0; p < n_dif.nchw(); p++) {
		n_dif(p) = 0;
		if (n_rsp(p) != mask_value) {
			n_dif(p) = 2 * inv_psize*(rsps->n_rsp(p) - n_rsp(p));
		}
	}
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
}
void sqr_error::backward_pass_mask(  float mask_value) {
	n_dif.resize(p_in1->n_rsp.size());
	all_error_for_batch = 0;
	float pcount = 0;
	for (int p = 0; p < n_dif.nchw(); p++) {
		if (n_rsp(p) != mask_value) {
			all_error_for_batch += double(p_in1->n_rsp(p) - n_rsp(p))*double(p_in1->n_rsp(p) - n_rsp(p));
			pcount++;
		}
	}
	avg_error = all_error_for_batch / double(pcount);
	float inv_psize = 1 / pcount;
	for (int p = 0; p < n_dif.nchw(); p++) {
		n_dif(p) = 0;
		if (n_rsp(p) != mask_value) {
			n_dif(p) = 2 * inv_psize*(p_in1->n_rsp(p) - n_rsp(p));
		}
	}
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
}

