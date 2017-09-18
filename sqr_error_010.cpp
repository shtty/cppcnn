////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#include "sqr_error_010.h"



sqr_error_010::sqr_error_010()
{
	n_name = "sqr_error";
	avg_error = 0;
	all_error_for_batch = 0;
}


sqr_error_010::~sqr_error_010()
{
}

double sqr_error_010::forward_pass() {
	all_error_for_batch = 0;
	double all_one_sum = 0;
	int nchw = p_in1->n_rsp.nchw();
	if ( nchw !=  n_rsp.nchw() ) {
		cout << "nchw of response and sqr_error_011 does not match, using smaller nchw..." << endl;
		if (nchw > n_rsp.nchw()) {
			nchw = n_rsp.nchw();
		}
	}
	for (int p = 0; p < nchw; p++) {
		all_error_for_batch += double(p_in1->n_rsp(p) - n_rsp(p))*double(p_in1->n_rsp(p) - n_rsp(p));
		all_one_sum += fabs(n_rsp(p));
	}
	avg_error = all_error_for_batch / double(all_one_sum);
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
	return avg_error;
}

double sqr_error_010::backward_pass() {
	n_dif.resize(p_in1->n_rsp.size());
	double all_one_sum = 0;
	for (int p = 0; p < n_rsp.nchw(); p++) {
		all_one_sum += fabs(n_rsp(p));
	}
	float inv_psize = 1 / float(all_one_sum);
	for (int p = 0; p < n_dif.nchw(); p++) {
		n_dif(p) = 2 * inv_psize*(p_in1->n_rsp(p) - n_rsp(p));
	}
	return avg_error;
}