#include "pool.h"


pool::pool(void)
{
	pool_type = "max";
	n_name = pool_type + "_pool";
}


pool::~pool(void)
{
}

void pool::load_init(ifstream &myfile, string layer_type ) {
	if (layer_type == "") {
		myfile >> layer_type;
	}
	//pool max 2 2 stride 2 pad 0 precious 0.000000
	string stemp;
	myfile >> pool_type;
	myfile >> n_pool.h;
	myfile >> n_pool.w;
	myfile >> stemp; // "stride"
	myfile >> n_stride.h;
	myfile >> n_stride.w;
	myfile >> stemp; // "pad"
	myfile >> n_pad.h;
	myfile >> n_pad.w;
}
void pool::save_init(ofstream &myfile) {
	myfile << endl;
	myfile<<"pool "<<pool_type<<" "<<n_pool.h<<" "<<n_pool.w<<" stride "<<n_stride.h<<" "<<n_stride.w<<" pad "<<n_pad.h<<" "<<n_pad.w<<endl;
}

double pool::forward_pass() {
	size4d isize = n_in1->n_rsp.size();
	isize.h /= n_stride.h;
	isize.w /= n_stride.w;
	n_rsp.resize(isize);
	n_backward_idx.resize(isize);
	//cout << "pool rsp size NCHW " << n_rsp.n() << " " << n_rsp.c() << " " << n_rsp.h() << " " << n_rsp.w() << endl;

	n_rsp.set(-FLT_MAX);
	isize = n_in1->n_rsp.size();

	for (int pn = 0; pn < isize.n; pn++) { // per each image
		for (int pc = 0; pc < isize.c; pc++) { // per each channel

			for (int rh = 0, ph = 0; rh < n_rsp.h(); rh++) {
				for (int rw = 0, pw = 0; rw < n_rsp.w(); rw++) {

					for (int mh = ph; mh < ph + n_pool.h && mh < isize.h; mh++) {
						for (int mw = pw; mw < pw + n_pool.w && mw < isize.w; mw++) {
							float in_rsp;
							in_rsp = n_in1->n_rsp(pn, pc, mh, mw);
							if (n_rsp(pn, pc, rh, rw) < in_rsp) {
								n_rsp(pn, pc, rh, rw) = in_rsp;
								// needed for backward pass only
								n_backward_idx(pn, pc, rh, rw) = float(n_in1->n_rsp.nchw2idx(pn, pc, mh, mw));
							}
						}
					}

					pw += n_stride.w;
				}
				ph += n_stride.h;
			}

		}
	}

	//float max_r = -FLT_MAX;
	//float min_r = FLT_MAX;
	//int tsize = n_rsp.nchw();
	//for (int p = 0; p < tsize; p++) {
	//	if (max_r < n_rsp(p)) { max_r = n_rsp(p); }
	//	if (min_r > n_rsp(p)) { min_r = n_rsp(p); }
	//}
	//cout << "max min: " << max_r << " " << min_r << endl;

	return 0;
}

double pool::backward_pass() {
	n_dif.resize(n_in1->n_rsp.size());
	n_dif.set(0);
	for (int p = 0; p < n_rsp.nchw(); p++) {
		n_dif(int(n_backward_idx(p))) += n_out1->n_dif(p);
	}
	return 0;
}