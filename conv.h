using namespace std;
#pragma once
#include "layer.h"
#ifdef SHTTY_CUDNN
#include "device_launch_parameters.h"
#include "C:\cudnn-8.0-windows10-x64-v5.1\cuda\include\cudnn.h"
#endif // SHTTY_CUDNN

class conv : public layer
{
private:
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
	void set_filter_NCHW(int n, int c, int h, int w, std::mt19937 rng = std::mt19937(0));
	void set_filter_NCHW(float min = 0.001f, float max = 0.001f, std::mt19937 rng = std::mt19937(0));
	void print(bool print_n_rsp = false);
	
	double forward_pass();
	void	forward_pass_cpu(layer *rsps);
	double backward_pass();
	void	backward_pass_cpu(layer *inlayer, layer *outlayer);
	void		update_gradient_cpu(layer *inlayer, layer *outlayer);
	void		update_bias(layer *outlayer );

	void	forward_pass_gpu(layer *rsps);
	void	backward_pass_gpu(layer *inlayer, layer *outlayer);
	void		update_gradient_gpu(layer *inlayer, layer *outlayer);

};