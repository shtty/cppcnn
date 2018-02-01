#pragma once
#include "conv.h"

class fully : public conv
{
public:
	string layer_type() { return "fully"; }

	fully();
	~fully();

	fully(const fully &cpy) : conv(cpy) {};
	fully& operator=(const fully &cpy);

	void load_init(ifstream &myfile, string layer_type = "");
	void save_init(ofstream &myfile);

	double forward_pass() {
		n_zero_padding = false;
		size4d rsize = p_in1->n_rsp.size();
		size4d fsize = n_weights.size();
		n_stride.h = rsize.h;
		n_stride.w = rsize.w;
		if ( rsize.c != fsize.c || rsize.h != fsize.h || rsize.w != fsize.w ) {
			cout << "fully: weight dim does not match!: possible error: setting weight dim to input rsp and xavier initialization" << endl;			
			n_weights_bias_set(fsize.n, rsize.c, rsize.h, rsize.w);
			
		}
		return conv::forward_pass();
	}
	double backward_pass(bool update_weights = true) {
		return conv::backward_pass(update_weights);
	}
};

