#include "residual.h"



residual::residual()
{
	n_relus.clear();
	n_n = n_c = n_h = n_w = 1;
	n_d = 0.01f;
	n_leak = 0;
	n_convs_size = 0;
}


residual::~residual()
{
	delete_p_convs();
	n_relus.clear();
}
residual::residual(const residual &cpy) : layer(cpy) {
	n_n = cpy.n_n; n_c = cpy.n_c; n_h = cpy.n_h; n_w = cpy.n_w;
	n_leak = cpy.n_leak;
	n_d = cpy.n_d;
	
	set_size(cpy.n_convs_size);

	//n_relus = cpy.n_relus;
	//
	//delete_p_convs();
	//n_convs_size = cpy.n_convs_size;
	//if (n_convs_size > 0) {
	//	p_convs = new conv[n_convs_size];
	//	for (int r = 0; r < n_convs_size; r++) {
	//		p_convs[r] = cpy.p_convs[r];
	//	}
	//}
	//set_links();
	n_name = cpy.n_name;
}
residual& residual::operator=(const residual &cpy) {
	n_name = cpy.n_name;
	n_use_gpu = cpy.n_use_gpu;
	p_in1 = cpy.p_in1;
	p_out1 = cpy.p_out1;
	n_rsp = cpy.n_rsp;
	n_dif = cpy.n_dif; // layer response, layer chain multiplier

	n_n = cpy.n_n; n_c = cpy.n_c; n_h = cpy.n_h; n_w = cpy.n_w;
	n_leak = cpy.n_leak;
	n_d = cpy.n_d;

	set_size(cpy.n_convs_size);

	//n_relus = cpy.n_relus;
	//delete_p_convs();
	//n_convs_size = cpy.n_convs_size;
	//if (n_convs_size > 0) {
	//	p_convs = new conv[n_convs_size];
	//	for (int r = 0; r < n_convs_size; r++) {
	//		p_convs[r] = cpy.p_convs[r];
	//	}
	//	set_links();
	//}
	n_name = cpy.n_name;
	
	return *this;
}

//void residual::set_output(residual &out_residual) {
//	n_converge.p_out1 = &out_residual.n_split;
//	out_residual.n_split.p_in1 = &n_converge;
//}
//void residual::set_output(layer &out_layer) {
//	n_converge.p_out1 = &out_layer;
//	out_layer.p_in1 = &n_converge;
//}
//void residual::set_input(residual &in_residual) {
//	n_split.p_in1 = &in_residual.n_converge;
//	in_residual.n_converge.p_out1 = &n_split;
//}
//void residual::set_input(layer &in_layer) {
//	n_split.p_in1 = &in_layer;
//	in_layer.p_out1 = &n_split;
//}


void residual::set_size(int size) {
	n_relus.clear();
	delete_p_convs();
	n_convs_size = size;
	if (n_convs_size > 0) {
		p_convs = new conv[n_convs_size];
		
		for (int r = 0; r < n_convs_size; r++) {
			relu ltemp;
			n_relus.push_back(ltemp);
		}
		n_weights_bias_set(n_n, n_c, n_h, n_w);
		set_links();
	}
}
void	residual::set_links() {
	n_split.p_in1 = p_in1;
	
	n_split.p_out1 = &p_convs[0];
	p_convs[0].p_in1 = &n_split;
	for (int r = 0; r < n_convs_size; r++) {
		p_convs[r].set_output(n_relus[r]);
	}
	for (int r = 0; r < n_convs_size-1; r++) {
		n_relus[r].set_output(p_convs[r + 1]);
	}
	n_split.p_out2 = &n_converge;
	n_converge.p_in2 = &n_split;
	n_converge.p_in1 = &n_relus[n_relus.size() - 1];
	n_relus[n_relus.size() - 1].p_out1 = &n_converge;

	n_converge.p_out1 = p_out1;
}
void	residual::n_weights_bias_set(int n, int c, int h, int w) {
	n_n = n; n_c = c; n_h = h; n_w = w;
	if (n_convs_size >= 1) {
		p_convs[0].n_weights_bias_set(n_n, n_c, n_h, n_w);
		p_convs[0].n_zero_padding = true;
		for (int r = 1; r <n_convs_size; r++) {
			p_convs[r].n_weights_bias_set(n_n, n_n, n_h, n_w);
			p_convs[r].n_zero_padding = true;
		}
		n_weights_set();
	}
}
void	residual::n_weights_set(string init_method, std::mt19937 &rng) {
	for (int r = 0; r < n_convs_size; r++) {
		p_convs[r].n_weights_set(init_method, rng);
	}
}
void residual::set_gradient_step_size(float d) {
	n_d = d;
	for (int r = 0; r < n_convs_size; r++) {
		p_convs[r].n_d = n_d;
	}
}

void residual::set_leak(float leak ) {
	n_leak = leak;
	for (int r = 0; r < n_relus.size(); r++) {
		n_relus[r].n_leak = n_leak;
	}
}

double residual::forward_pass() {
	set_links();
	n_split.forward_pass();
	for (int r = 0; r < n_convs_size; r++) {
		p_convs[r].forward_pass();
		n_relus[r].forward_pass();
	}
	n_converge.forward_pass();
	n_rsp = n_converge.n_rsp;
	return 0;
}
double residual::backward_pass(bool update_weights) {
	set_links();
	n_converge.backward_pass(update_weights);
	for (int r = n_convs_size - 1; r >= 0; r--) {
		n_relus[r].backward_pass(update_weights);
		p_convs[r].backward_pass(update_weights);
	}
	n_split.backward_pass(update_weights);
	n_dif = n_split.n_dif;
	return 0;
}