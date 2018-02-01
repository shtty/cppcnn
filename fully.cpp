#include "fully.h"



fully::fully()
{
	n_name = "fully";
	n_d = 0.01f;
}
fully::~fully()
{
}
fully& fully::operator=(const fully &cpy) {
	n_name = cpy.n_name;
	n_use_gpu = cpy.n_use_gpu;
	n_d = cpy.n_d;
	p_in1 = cpy.p_in1;
	p_out1 = cpy.p_out1;
	n_rsp = cpy.n_rsp;
	n_dif = cpy.n_dif; // layer response, layer chain multiplier

	n_weights = cpy.n_weights;
	n_bias = cpy.n_bias;
	n_bias_gradient = cpy.n_bias_gradient;
	n_weights_gradient = cpy.n_weights_gradient;

	return *this;
}
void fully::load_init(ifstream &myfile, string layer_type) {
	if (layer_type == "") {
		myfile >> layer_type;
	}
	string stemp;

	int itemp;
	myfile >> stemp; // "weight_dim"
	myfile >> itemp; // 4
	myfile >> stemp; // "size_NCHW"

	int N, C, H, W;
	myfile >> N;
	myfile >> C;
	myfile >> H;
	myfile >> W;
	n_weights.resize(N, C, H, W);
	n_bias.resize(N);
}
void fully::save_init(ofstream &myfile) {
	myfile << endl;
	myfile << "fully weight_dim " << 4 << " size_NCHW " << n_weights.n() << " " << n_weights.c() << " " << n_weights.h() << " " << n_weights.w() << " " << endl;
}


//void fully::load_weights(ifstream &myfile) {
//	for (int n = 0; n < n_weights.n(); n++) {
//		for (int c = 0; c < n_weights.c(); c++) {
//			for (int h = 0; h < n_weights.h(); h++) {
//				for (int w = 0; w < n_weights.w(); w++) {
//					myfile >> n_weights(n, c, h, w);
//				}
//			}
//		}
//	}
//	for (int n = 0; n < n_bias.size(); n++) {
//		myfile >> n_bias(n);
//	}
//	int stop = 1;
//}
//void fully::save_weights(ofstream &myfile) {
//	myfile << endl << std::scientific;
//	for (int n = 0; n < n_weights.n(); n++) {
//		for (int c = 0; c < n_weights.c(); c++) {
//			for (int h = 0; h < n_weights.h(); h++) {
//				for (int w = 0; w < n_weights.w(); w++) {
//					myfile << n_weights(n, c, h, w) << " ";
//				}
//				myfile << endl;
//			}
//			myfile << endl;
//		}
//	}
//	for (int n = 0; n < n_bias.size(); n++) {
//		myfile << n_bias(n) << endl;
//	}
//	myfile << endl;
//}
//void fully::n_weights_bias_set(int n, int c, int h, int w) {
//	n_weights.resize(n, c, h, w);
//	n_weights.set("xavier");
//	n_bias.resize(n);
//	n_bias.set(0);
//}
//void fully::n_weights_set(string init_method, std::mt19937 &rng) {
//	n_weights.set(init_method, rng);
//}
//void fully::print(bool print_n_rsp) {
//	layer::print(print_n_rsp);
//	cout << "n_weights ";
//	n_weights.print(print_n_rsp);
//	cout << "n_bias ";
//	n_bias.print(print_n_rsp);
//}

//double fully::forward_pass() {
//	if (n_use_gpu) {
//		forward_pass_gpu(p_in1);
//	}
//	else {
//		forward_pass_cpu(p_in1);
//	}
//	return 0;
//}
//void	fully::forward_pass_cpu(layer *rsps) {
//	if ( rsps->n_rsp.chw() != n_weights.chw()) {
//		cout << "fully: input response and weight chw size does not match!" << endl;
//		int filter_number = n_weights.n();
//		n_weights.resize( filter_number, rsps->n_rsp.c(), rsps->n_rsp.h(), rsps->n_rsp.w() );
//	}
//	n_rsp.resize( rsps->n_rsp.n() , n_weights.n() , 1, 1);
//	
//	n_rsp.set(0);
//	int chw = n_weights.chw();
//	for (int n = 0; n < n_rsp.n() ; n++) {
//	for (int c = 0; c < n_rsp.c() ; c++) {
//		for (int p = 0; p < chw ; p++) {
//			n_rsp(n, c, 0, 0) += n_weights(c, p)*(rsps->n_rsp(n,p));
//		}
//	}}
//	for (int n = 0; n < n_rsp.n(); n++) {
//		for (int c = 0; c < n_rsp.c(); c++) {
//			n_rsp(n, c, 0, 0) += n_bias(c);
//		}
//	}
//}
//void	fully::forward_pass_gpu(layer *rsps) {
//#ifdef SHTTY_CUDNN
//	forward_pass_cpu(rsps);
//#else
//	forward_pass_cpu(rsps);
//#endif
//}
//double fully::backward_pass() {
//	if (n_use_gpu) {
//		backward_pass_gpu(p_in1, p_out1);
//	}
//	else {
//		backward_pass_cpu(p_in1, p_out1);
//	}
//	update_bias(p_out1);
//
//	////// set max and min gradients 
//	std::uniform_real_distribution<float> uniform_dist(0, 1);
//	for (int p = 0; p < n_weights.nchw(); p++) {
//		if (n_d*n_weights_gradient(p) > 1) {
//			n_weights_gradient(p) =  uniform_dist(float4d::n_random_seed) / n_d;
//		}
//		if (n_d*n_weights_gradient(p) < -1) {
//			n_weights_gradient(p) = -uniform_dist(float4d::n_random_seed) / n_d;
//		}
//	}
//	////// update filter weights
//	for (int p = 0; p < n_weights.nchw(); p++) {
//		n_weights(p) = n_weights(p) - n_d*n_weights_gradient(p);
//	}
//	return 0;
//}
//void	fully::backward_pass_cpu(layer *inlayer, layer *outlayer) {
//	n_dif.resize(inlayer->n_rsp.size());
//	
//	n_dif.set(0);
//	int chw = n_weights.chw();
//	for (int n = 0; n < n_rsp.n(); n++) { // # of samples
//	for (int c = 0; c < n_rsp.c(); c++) { // # of filters == c of output response == chw of output response
//		for (int p = 0; p < chw; p++) {   // chw of input rsp == chw of filter rsp
//			n_dif(n, p) += outlayer->n_dif(n, c, 0, 0) * n_weights(c, p);
//		}
//	}}
//	update_gradient_cpu(inlayer, outlayer);
//}
//void		fully::update_gradient_cpu(layer *inlayer, layer *outlayer) {
//	n_weights_gradient.resize(n_weights.size());
//	n_weights_gradient.set(0);
//	int chw = n_weights.chw();
//	for (int n = 0; n < n_rsp.n(); n++) { // # of samples
//	for (int c = 0; c < n_rsp.c(); c++) { // # of filters == c of output response == chw of output response
//		for (int p = 0; p < chw; p++) {   // chw of input rsp == chw of filter rsp
//			n_weights_gradient(c, p) += outlayer->n_dif(n, c) * inlayer->n_rsp(n, p);
//		}
//	}}
//}
//void	fully::backward_pass_gpu(layer *inlayer, layer *outlayer) {
//#ifdef SHTTY_CUDNN
//	backward_pass_cpu(inlayer, outlayer);
//#else
//	backward_pass_cpu(inlayer, outlayer);
//#endif
//}
//void	fully::update_bias(layer *outlayer) {
//	////// Calculate gradient for bias
//	n_bias_gradient.resize(n_bias.size());
//	n_bias_gradient.set(0);
//	for (int pn = 0; pn < n_rsp.n(); pn++) { // per each image sample
//	for (int pc = 0; pc < n_rsp.c(); pc++) { // c of out response = chw of out response	
//		n_bias_gradient(pc) += outlayer->n_dif(pn, pc, 0, 0);
//	}}
//
//	std::uniform_real_distribution<float> uniform_dist(0, 1);
//	for (int p = 0; p < n_bias_gradient.size(); p++) {
//		if (n_d*n_bias_gradient(p) > 1) {
//			n_bias_gradient(p) = uniform_dist(float4d::n_random_seed) / n_d;
//		}
//		if (n_d*n_bias_gradient(p) < -1) {
//			n_bias_gradient(p) = -uniform_dist(float4d::n_random_seed) / n_d;
//		}
//	}
//	//// update bias 
//	for (int p = 0; p < n_bias.size(); p++) {
//		n_bias(p) = n_bias(p) - n_d*n_bias_gradient(p);
//	}
//}