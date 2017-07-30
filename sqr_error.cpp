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
	return backward_pass();
}

void sqr_error::backward_pass(layer *rsps) {
	double cpu_error = 0, gpu_error = 0;
	avg_error = backward_pass_cpu(rsps);
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
	cout << std::scientific;
}
double sqr_error::backward_pass() {
	double cpu_error = 0, gpu_error = 0;
	avg_error = backward_pass_cpu(n_in1);
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
	return avg_error;
}
double sqr_error::backward_pass_cpu(layer *rsps) {

	n_dif.resize(rsps->n_rsp.size());
	all_error_for_batch = 0;
	for (int p = 0; p < n_dif.nchw(); p++) {
		all_error_for_batch += double(rsps->n_rsp(p) - n_rsp(p) )*double(rsps->n_rsp(p) - n_rsp(p));
	}
	avg_error = all_error_for_batch / double(n_dif.nchw());

	float inv_psize = 1 / float(n_dif.nchw());
	for (int p = 0; p < n_dif.nchw(); p++) {
		n_dif(p) = 2*inv_psize*(rsps->n_rsp(p) - n_rsp(p));
	}
	
	//cout << "All Error: "  << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	//cout <<	"  Avg Error: "<< std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
	//cout << "Error partial diff size NCHW: " << n_dif.n() << " " << n_dif.c() << " " << n_dif.h() << " " << n_dif.w() << endl;

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
	n_dif.resize(n_in1->n_rsp.size());
	all_error_for_batch = 0;
	float pcount = 0;
	for (int p = 0; p < n_dif.nchw(); p++) {
		if (n_rsp(p) != mask_value) {
			all_error_for_batch += double(n_in1->n_rsp(p) - n_rsp(p))*double(n_in1->n_rsp(p) - n_rsp(p));
			pcount++;
		}
	}
	avg_error = all_error_for_batch / double(pcount);
	float inv_psize = 1 / pcount;
	for (int p = 0; p < n_dif.nchw(); p++) {
		n_dif(p) = 0;
		if (n_rsp(p) != mask_value) {
			n_dif(p) = 2 * inv_psize*(n_in1->n_rsp(p) - n_rsp(p));
		}
	}
	cout << "All Error: " << std::fixed << std::setw(11) << std::setprecision(6) << all_error_for_batch;
	cout << "  Avg Error: " << std::fixed << std::setw(11) << std::setprecision(6) << avg_error << "\xd"; // endl;
}

