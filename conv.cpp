#include "conv.h"

conv::conv(void)
{
	n_name = "conv";
	n_pad.h = 0;
	n_pad.w = 0;
	n_stride.h = 1;
	n_stride.w = 1;
	n_d = 0.01f;
#ifdef SHTTY_CUDNN
	cudnnCreate(                     &gpu_handle);
	cudnnCreateFilterDescriptor(     &gpu_filterDesc);
	cudnnCreateTensorDescriptor(     &gpu_InputDesc);
	cudnnCreateTensorDescriptor(     &gpu_OutputDesc);
	cudnnCreateConvolutionDescriptor(&gpu_convDesc);
	n_filter_mem_size = sizeof(float);
	n_input_mem_size  = sizeof(float);
	n_output_mem_size = sizeof(float);
	cudaMalloc((void**)&dev_filter_mem, n_filter_mem_size);
	cudaMalloc((void**)&dev_input_mem,  n_input_mem_size );
	cudaMalloc((void**)&dev_output_mem, n_output_mem_size);	
	n_workspace_fw_size = sizeof(float);
	n_workspace_bw_size = sizeof(float);
	n_workspace_wt_size = sizeof(float);
	cudaMalloc((void**)&dev_workspace_fw, n_workspace_fw_size);
	cudaMalloc((void**)&dev_workspace_bw, n_workspace_bw_size);
	cudaMalloc((void**)&dev_workspace_wt, n_workspace_wt_size);
#endif // SHTTY_CUDNN
}
conv::~conv(void)
{
#ifdef SHTTY_CUDNN
	cudnnDestroy(gpu_handle);
	cudnnDestroyFilterDescriptor(gpu_filterDesc);
	cudnnDestroyTensorDescriptor(gpu_InputDesc);
	cudnnDestroyTensorDescriptor(gpu_OutputDesc);
	cudnnDestroyConvolutionDescriptor(gpu_convDesc);
	cudaFree(dev_filter_mem);
	cudaFree(dev_input_mem);
	cudaFree(dev_output_mem);
	cudaFree(dev_workspace_fw);
	cudaFree(dev_workspace_bw);
	cudaFree(dev_workspace_wt);
#endif // SHTTY_CUDNN
}
conv::conv(const conv &cpy) : layer(cpy) {
	n_weights = cpy.n_weights;
	n_bias = cpy.n_bias;
	n_bias_gradient = cpy.n_bias_gradient;
	n_weights_gradient = cpy.n_weights_gradient;	
	conv();
}
conv& conv::operator=(const conv &cpy) {
	n_name = cpy.n_name;
	n_use_gpu = cpy.n_use_gpu;
	n_d = cpy.n_d;
	n_in1 = cpy.n_in1;
	n_out1 = cpy.n_out1;
	//leak; n_pool;
	n_stride = cpy.n_stride;
	n_pad = cpy.n_pad;
	n_rsp = cpy.n_rsp;
	n_dif = cpy.n_dif; // layer response, layer chain multiplier

	n_weights = cpy.n_weights;
	n_bias = cpy.n_bias;
	n_bias_gradient = cpy.n_bias_gradient;
	n_weights_gradient = cpy.n_weights_gradient;

	return *this;
}

void conv::load_init(ifstream &myfile, string layer_type) {
	if ( layer_type == "" ) {
		myfile >> layer_type;
	}
	string stemp;
	myfile >> stemp; // "stride"
	myfile >> n_stride.h;
	myfile >> n_stride.w;
	myfile >> stemp; // "pad"
	myfile >> n_pad.h;
	myfile >> n_pad.w;

	int itemp;
	myfile >> stemp; // "weight_dim"
	myfile >> itemp;
	myfile >> stemp; // "size_NCHW"
	
	int N, C, H, W;
	myfile >> N;
	myfile >> C;
	myfile >> H;
	myfile >> W;
	n_weights.resize(N, C, H, W);
	n_bias.resize(N);
}
void conv::save_init(ofstream &myfile) {
	myfile << endl ;
	myfile << "conv stride " << n_stride.h << " " << n_stride.w << " pad " << n_pad.h << " " << n_pad.w << endl;
	myfile << "weight_dim " << 4 << " size_NCHW " << n_weights.n() << " " << n_weights.c() << " " << n_weights.h() << " " << n_weights.w() << endl;

}
void conv::n_weights_bias_set(int n, int c, int h, int w) {
	n_weights.resize(n,c,h,w);
	n_weights.set("xavier");
	n_bias.resize(n);
	n_bias.set(0);
}
void conv::n_weights_set(string init_method, std::mt19937 &rng) {
	n_weights.set(init_method, rng);
}
void conv::load_weights(ifstream &myfile) {
	
	for (int n = 0; n < n_weights.n(); n++) {
	for (int c = 0; c < n_weights.c(); c++) {
	for (int h = 0; h < n_weights.h(); h++) {
	for (int w = 0; w < n_weights.w(); w++) {
		myfile >> n_weights(n, c, h, w);
	}}}}
	for (int n = 0; n < n_bias.size(); n++ ) {
		myfile >> n_bias(n);
	}
	int stop = 1;
}
void conv::save_weights(ofstream &myfile) {
	myfile << endl << std::scientific;
	for (int n = 0; n < n_weights.n(); n++) {
		for (int c = 0; c < n_weights.c(); c++) {
			for (int h = 0; h < n_weights.h(); h++) {
				for (int w = 0; w < n_weights.w(); w++) {
					myfile << n_weights(n, c, h, w) << " ";
				}
				myfile << endl;
			}
			myfile << endl;
		}
	}
	for (int n = 0; n < n_bias.size(); n++) {
		myfile << n_bias(n) << endl ;
	}
	myfile << endl;
}
void conv::print( bool print_n_rsp) {
	layer::print(print_n_rsp);
	cout << "n_weights ";
	n_weights.print(print_n_rsp );
	cout << "n_bias ";
	n_bias.print(print_n_rsp);
}

double conv::forward_pass() {
	if (n_use_gpu) {
		forward_pass_gpu(n_in1);
	}
	else {
		forward_pass_cpu(n_in1);
	}
	return 0;
}
void	conv::forward_pass_cpu(layer *rsps) {
	size4d rsize;
	rsize = rsps->n_rsp.size();
	int pad_h = n_weights.h() / 2;
	int pad_w = n_weights.w() / 2;
	if (n_pad.h == 0) {
		rsize.h -= (n_weights.h() - 1);
		pad_h = 0;
	}
	if (n_pad.w == 0) {
		rsize.w -= (n_weights.w() - 1);
		pad_w = 0;
	}
	if ( rsize.h % n_stride.h > 0) { rsize.h = (rsize.h / n_stride.h) + 1; }
	else { rsize.h /= n_stride.h; }
	if ( rsize.w % n_stride.w > 0) { rsize.w = (rsize.w / n_stride.w) + 1; }
	else { rsize.w /= n_stride.w; }
	rsize.c = n_weights.n();

	n_rsp.resize(rsize);
	n_rsp.set(0);

	for (int pn = 0; pn < n_rsp.n() ; pn++) { // per each image sample
	for (int pc = 0; pc < n_rsp.c() ; pc++) { // per each channel
	for (int ph = 0; ph < n_rsp.h() ; ph++) { // per each y
	for (int pw = 0; pw < n_rsp.w() ; pw++) { // per each x

		for (int wc = 0; wc < n_weights.c() ; wc++) {
		for (int wh = 0; wh < n_weights.h() ; wh++) {
		for (int ww = 0; ww < n_weights.w() ; ww++) {

			float rsp_value = rsps->n_rsp.at( pn, wc, (ph*n_stride.h + wh - pad_h), (pw*n_stride.w + ww - pad_w) );
			n_rsp(pn, pc, ph, pw) += rsp_value * n_weights(pc, wc, n_weights.h() - 1 - wh, n_weights.w() - 1 - ww);

		}}}

	}}}}	
	// add bias....
	for (int pn = 0; pn < n_rsp.n(); pn++) { // per each image sample
	for (int pc = 0; pc < n_rsp.c(); pc++) { // per each channel
	for (int ph = 0; ph < n_rsp.h(); ph++) { // per each y
	for (int pw = 0; pw < n_rsp.w(); pw++) { // per each x
		n_rsp(pn, pc, ph, pw) += n_bias(pc);
	}}}}
	int stop = 1;
	
}
double conv::backward_pass() {
	if (n_use_gpu) {
		backward_pass_gpu(n_in1, n_out1);
	}
	else {
		backward_pass_cpu(n_in1, n_out1);
	}
	update_bias(n_out1);
	
	////// set max and min gradients 
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	//std::uniform_real_distribution<float> uniform_dist2(-1, 1);
	for (int p = 0; p < n_weights.nchw(); p++) {
		if ( n_d*n_weights_gradient(p) > 1 ) {
			n_weights_gradient(p) = uniform_dist(float4d::n_random_seed) / n_d;
		}
		if (n_d*n_weights_gradient(p) < -1) {
			n_weights_gradient(p) = -uniform_dist(float4d::n_random_seed) / n_d;
		}
		//if ( fabs(n_weights_gradient(p)) < 0.25f ) {
		//	n_weights_gradient(p) = float(uniform_dist2(n_random_seed))*0.25f;
		//}
	}
	////// update filter weights
	for (int p = 0; p < n_weights.nchw(); p++) {
		n_weights(p) = n_weights(p) - n_d*n_weights_gradient(p);
	}
	return 0;
}
void		conv::update_gradient_cpu(layer *inlayer, layer *outlayer) {
	//// calculate gradients for linear filter weights
	int stride_h, stride_w;
	stride_h = int(n_stride.h);
	stride_w = int(n_stride.w);
	int pad_h = n_weights.h() / 2;
	int pad_w = n_weights.w() / 2;
	if (n_pad.h == 0) { pad_h = 0; }
	if (n_pad.w == 0) { pad_w = 0; }
	n_weights_gradient.resize(n_weights.size());
	n_weights_gradient.set(0);

	for (int pn = 0; pn < n_rsp.n(); pn++) { // per each image sample
	for (int pc = 0; pc < n_rsp.c(); pc++) { // per each channel
	for (int ph = 0; ph < n_rsp.h(); ph++) { // per each y
	for (int pw = 0; pw < n_rsp.w(); pw++) { // per each x

		for (int wc = 0; wc < n_weights.c(); wc++) {
		for (int wh = 0; wh < n_weights.h(); wh++) {
		for (int ww = 0; ww < n_weights.w(); ww++) {

			int rsp_h = ph*stride_h + wh - pad_h;
			int rsp_w = pw*stride_w + ww - pad_w;
			float rsp_value = inlayer->n_rsp.at(pn, wc, rsp_h, rsp_w);
			n_weights_gradient(pc, wc, n_weights.h() - 1 - wh, n_weights.w() - 1 - ww) += (outlayer->n_dif(pn, pc, ph, pw))*rsp_value;
			//n_weights_gradient(pc, wc, wh, ww) += (outlayer->n_dif(pn, pc, ph, pw))*rsp_value;

		}}}

	}}}}

}
void	conv::backward_pass_cpu(layer *inlayer, layer *outlayer) {
	int pad_h = n_weights.h() / 2;
	int pad_w = n_weights.w() / 2;
	if (n_pad.h == 0) { pad_h = 0; }
	if (n_pad.w == 0) { pad_w = 0; }
	
	n_dif.resize( inlayer->n_rsp.size() );
	n_dif.set(0);
	// bias does not contribute to the calculation of n_diff

	for (int pn = 0; pn < n_rsp.n() ; pn++) { // per each image sample
	for (int pc = 0; pc < n_rsp.c() ; pc++) { // per each channel
	for (int ph = 0; ph < n_rsp.h() ; ph++) { // per each y
	for (int pw = 0; pw < n_rsp.w() ; pw++) { // per each y

		for (int wc = 0; wc < n_weights.c() ; wc++) {
		for (int wh = 0; wh < n_weights.h() ; wh++) {
		for (int ww = 0; ww < n_weights.w() ; ww++) {
			//float rsp_value = rsps->n_rsp(pn, wc, (ph*n_stride.h + wh - pad_h), (pw*n_stride.w + ww - pad_w));
			//n_rsp(pn, pc, ph, pw) += rsp_value * n_weights(pc, wc, wh, ww);
			int dif_h = ph*n_stride.h + wh - pad_h;
			int dif_w = pw*n_stride.w + ww - pad_w;
			if (dif_h >= 0 && dif_h < n_dif.h() && dif_w >= 0 && dif_w < n_dif.w() ) {
				n_dif(pn, wc, dif_h, dif_w) += outlayer->n_dif(pn, pc, ph, pw) * n_weights(pc, wc, n_weights.h() - 1 - wh, n_weights.w() - 1 - ww);
			}

		}}}

	}}}}
	
	update_gradient_cpu(inlayer, outlayer);
}
void	conv::update_bias(layer *outlayer ) {
	//// Calculate gradient for bias
	n_bias_gradient.resize(n_bias.size());
	n_bias_gradient.set(0);
	for (int pn = 0; pn < n_rsp.n(); pn++) { // per each image sample
	for (int pc = 0; pc < n_rsp.c(); pc++) { // per each channel		
	for (int ph = 0; ph < n_rsp.h(); ph++) { // per each y
	for (int pw = 0; pw < n_rsp.w(); pw++) { // per each x
		n_bias_gradient(pc) += outlayer->n_dif(pn, pc, ph, pw);
	}}}}
	
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	for (int p = 0; p < n_bias_gradient.size(); p++) {
		if (n_d*n_bias_gradient(p) > 1) {
			n_bias_gradient(p) = uniform_dist(float4d::n_random_seed) / n_d;
		}
		if (n_d*n_bias_gradient(p) < -1) {
			n_bias_gradient(p) = -uniform_dist(float4d::n_random_seed) / n_d;
		}
	}
	// update bias 
	for (int p = 0; p < n_bias.size(); p++) {
		n_bias(p) = n_bias(p) - n_d*n_bias_gradient(p);
	}
}

void conv::forward_pass_gpu(layer *rsps) {
	//// GPU forward pass match perfectly with CPU forward pass....
	//// Unless it is even sized filters
#ifdef SHTTY_CUDNN
	int pad_h = n_weights.h() / 2;
	int pad_w = n_weights.w() / 2;
	if (n_pad.h == 0) { pad_h = 0; }
	if (n_pad.w == 0) { pad_w = 0; }

	int IW, IH, IC, IN;
	int FW, FH, FC, FN;
	IW = rsps->n_rsp.w();
	IH = rsps->n_rsp.h();
	IC = rsps->n_rsp.c();
	IN = rsps->n_rsp.n();

	FW = n_weights.w();
	FH = n_weights.h();
	FC = n_weights.c();
	FN = n_weights.n();

	cudnnStatus_t cudnn_status_check;

	cudnn_status_check = cudnnSetTensor4dDescriptor(gpu_InputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, IN, IC, IH, IW);
	cudnn_status_check = cudnnSetFilter4dDescriptor(gpu_filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, FN, FC, FH, FW);
	int upscale_x = 1, upscale_y = 1;
	cudnn_status_check = cudnnSetConvolution2dDescriptor(gpu_convDesc, pad_h, pad_w, int(n_stride.h), int(n_stride.w), upscale_x, upscale_y, CUDNN_CONVOLUTION);

	int ON, OC, OH, OW;
	cudnn_status_check = cudnnGetConvolution2dForwardOutputDim(gpu_convDesc, gpu_InputDesc, gpu_filterDesc, &ON, &OC, &OH, &OW);
	cudnn_status_check = cudnnSetTensor4dDescriptor(gpu_OutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ON, OC, OH, OW);
	n_rsp.resize(ON, OC, OH, OW);

	size_t sizeInBytes = 0;
	cudnn_status_check = cudnnGetConvolutionForwardWorkspaceSize(gpu_handle, gpu_InputDesc, gpu_filterDesc, gpu_convDesc, gpu_OutputDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &sizeInBytes);

	int newsize;
	newsize = sizeInBytes;
	if (newsize != n_workspace_fw_size) {
		n_workspace_fw_size = newsize;
		cudaFree(dev_workspace_fw);
		cudaMalloc((void**)&dev_workspace_fw, n_workspace_fw_size);
	}
	newsize = IN*IC*IW*IH * sizeof(float);
	if (newsize != n_input_mem_size) {
		n_input_mem_size = newsize;
		cudaFree(dev_input_mem);
		cudaMalloc((void**)&dev_input_mem, n_input_mem_size);
	}
	newsize = FN*FC*FW*FH * sizeof(float);
	if (newsize != n_filter_mem_size) {
		n_filter_mem_size = newsize;
		cudaFree(dev_filter_mem);
		cudaMalloc((void**)&dev_filter_mem, n_filter_mem_size);
	}
	newsize = ON*OC*OW*OH * sizeof(float);
	if (newsize != n_output_mem_size) {
		n_output_mem_size = newsize;
		cudaFree(dev_output_mem);
		cudaMalloc((void**)&dev_output_mem, n_output_mem_size);
	}

	cudaMemcpy(dev_input_mem, &(rsps->n_rsp(0)), n_input_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_filter_mem, &(n_weights(0)), n_filter_mem_size, cudaMemcpyHostToDevice);;
	float alpha = 1, beta = 0;
	cudnn_status_check = cudnnConvolutionForward(gpu_handle, &alpha, gpu_InputDesc, dev_input_mem, gpu_filterDesc, dev_filter_mem, gpu_convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, dev_workspace_fw, sizeInBytes, &beta, gpu_OutputDesc, dev_output_mem);
	cudaMemcpy(&(n_rsp(0)), dev_output_mem, n_output_mem_size, cudaMemcpyDeviceToHost);

	// add bias....
	for (int pn = 0; pn < n_rsp.n(); pn++) { // per each image sample
		for (int pc = 0; pc < n_rsp.c(); pc++) { // per each channel
			for (int ph = 0; ph < n_rsp.h(); ph++) { // per each y
				for (int pw = 0; pw < n_rsp.w(); pw++) { // per each x
					n_rsp(pn, pc, ph, pw) += n_bias(pc);
				}
			}
		}
	}
#else
	forward_pass_cpu(rsps);
#endif // SHTTY_CUDNN
}
void conv::backward_pass_gpu(layer *inlayer, layer *outlayer) {
#ifdef SHTTY_CUDNN
	n_dif.resize(inlayer->n_rsp.size());
	cudnnStatus_t cudnn_status_check;
	size_t sizeInBytes = 0;
	cudnn_status_check = cudnnGetConvolutionBackwardDataWorkspaceSize(gpu_handle, gpu_filterDesc, gpu_OutputDesc, gpu_convDesc, gpu_InputDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &sizeInBytes);
	if (sizeInBytes != n_workspace_bw_size) {
		n_workspace_bw_size = sizeInBytes;
		cudaFree(dev_workspace_bw);
		cudaMalloc((void**)&dev_workspace_bw, n_workspace_bw_size);
	}
	cudaMemcpy(dev_input_mem, &(n_dif(0)), n_input_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_output_mem, &(outlayer->n_dif(0)), n_output_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_filter_mem, &(n_weights(0)), n_filter_mem_size, cudaMemcpyHostToDevice);
	float alpha = 1, beta = 0;
	cudnn_status_check = cudnnConvolutionBackwardData(gpu_handle, &alpha, gpu_filterDesc, dev_filter_mem, gpu_OutputDesc, dev_output_mem, gpu_convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, dev_workspace_bw, sizeInBytes, &beta, gpu_InputDesc, dev_input_mem);
	cudaMemcpy(&(n_dif(0)), dev_input_mem, n_input_mem_size, cudaMemcpyDeviceToHost);

	///////////////////////// update gradient
	update_gradient_gpu(inlayer, outlayer);
#else
	backward_pass_cpu(inlayer, outlayer);
#endif //SHTTY_CUDNN
}
void	conv::update_gradient_gpu(layer *inlayer, layer *outlayer) {
#ifdef SHTTY_CUDNN
	n_weights_gradient.resize( n_weights.size() );
	n_weights_gradient.set(0);
	
	cudnnStatus_t cudnn_status_check;
	size_t sizeInBytes = 0;
	cudnn_status_check = cudnnGetConvolutionBackwardFilterWorkspaceSize(gpu_handle, gpu_InputDesc, gpu_OutputDesc, gpu_convDesc, gpu_filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, &sizeInBytes);
	if (sizeInBytes != n_workspace_wt_size ) {
		n_workspace_wt_size = sizeInBytes;
		cudaFree(dev_workspace_wt);
		cudaMalloc((void**)&dev_workspace_wt, n_workspace_wt_size);
	}
	
	cudaMemcpy(dev_input_mem, &( inlayer->n_rsp(0)), n_input_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_output_mem, &(outlayer->n_dif(0)), n_output_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_filter_mem, &(n_weights(0)), n_filter_mem_size, cudaMemcpyHostToDevice);
	float alpha = 1, beta = 0;
	cudnn_status_check = cudnnConvolutionBackwardFilter(gpu_handle, &alpha, gpu_InputDesc, dev_input_mem, gpu_OutputDesc, dev_output_mem, gpu_convDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, dev_workspace_wt, sizeInBytes, &beta, gpu_filterDesc, dev_filter_mem);
	cudaMemcpy(&(n_weights_gradient(0)), dev_filter_mem, n_filter_mem_size, cudaMemcpyDeviceToHost);

#else
	update_gradient_cpu(inlayer, outlayer);
#endif
}

