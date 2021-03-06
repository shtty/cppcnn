////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
using namespace std;
#pragma once
#include "layer.h"
#ifdef SHTTY_CUDNN
#include "device_launch_parameters.h"
#include "C:\cudnn-8.0-windows10-x64-v5.1\cuda\include\cudnn.h"
// Properties->Linker->general->Additional Library Directories
// C:\cudnn - 8.0 - windows10 - x64 - v5.1\cuda\lib\x64
// Properties->Linker->Input->Additional Dependencies
// cudnn.lib
// cudart.lib
// kernel32.lib
// copy cudnn64_5.dll onto folder with exe file
#endif // SHTTY_CUDNN

class conv : public layer
{
private:
	static int n_conv_count;
#ifdef SHTTY_CUDNN
	cudnnHandle_t                gpu_handle;
	cudnnFilterDescriptor_t	     gpu_filterDesc;
	cudnnTensorDescriptor_t	     gpu_InputDesc;
	cudnnTensorDescriptor_t	     gpu_OutputDesc;
	cudnnConvolutionDescriptor_t gpu_convDesc;
	float *dev_filter_mem;
	float *dev_input_mem;
	float *dev_output_mem;
	int n_filter_mem_size, n_input_mem_size, n_output_mem_size;

	float *dev_workspace_fw;   int n_workspace_fw_size;
	float *dev_workspace_bw;   int n_workspace_bw_size;
	float *dev_workspace_wt;   int n_workspace_wt_size;
#endif // SHTTY_CUDNN
protected:
public:
	string layer_type() { return "conv"; }
	float   n_d;    // learning rate gradient step size
	bool	n_zero_padding;
	size2d  n_stride;
	float4d n_weights;
	float1d n_bias;
	float1d n_bias_gradient;
	float4d n_weights_gradient;

	conv(void);
	~conv(void);
	conv(const conv &cpy);
	conv& operator=(const conv &cpy);
	
	///// Even Sized filters: There will be size discrepencies between CPU and GPU imlementation
	///// Must fix CPU or GPU forward and backward
	void load_init(ifstream &myfile, string layer_type = "" );
	void save_init(ofstream &myfile);
	void load_weights(ifstream &myfile);
	void save_weights(ofstream &myfile);
	void n_weights_bias_set(int n, int c, int h, int w );
	void n_weights_set(string init_method = "xavier", std::mt19937 &rng = float4d::n_random_seed);
	void set_learning_rate(float d) { n_d = d; }
	void print(bool print_n_rsp = false);
	
	double forward_pass();
	void	forward_pass_cpu(layer *rsps);
	double backward_pass(bool update_weights = true);
	void	backward_pass_cpu(layer *inlayer, layer *outlayer);
	void		update_gradient_cpu(layer *inlayer, layer *outlayer);
	void		update_bias(layer *outlayer );

	void	forward_pass_gpu(layer *rsps);
	void	backward_pass_gpu(layer *inlayer, layer *outlayer);
	void		update_gradient_gpu(layer *inlayer, layer *outlayer);

};
