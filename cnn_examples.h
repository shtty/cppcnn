#pragma once
#include "linear_cnn.h"

void cnn_example_using_layer();
void cnn_example_using_linear_cnn();

void cnn_example_using_layer() {
	// a simple generative model test
	layer data1;
	conv  conv1;
	pool  pool1;
	relu  relu1;
	c4to1 upsc1;
	conv  conv2;
	sqr_error error;

	// set in and out of layers
	conv1.setinput(&data1);
	pool1.setinput(&conv1);
	relu1.setinput(&pool1);
	upsc1.setinput(&relu1);
	conv2.setinput(&upsc1);
	error.setinput(&conv2);

	// set parameters on the layers
	// some parameters are default values at the constructor
	data1.n_rsp.setsize(2, 2, 8, 8);
	data1.n_rsp.set(0.5f);
	data1.n_rsp.print();

	conv1.set_filter_NCHW(4, 2, 3, 3);
	conv1.n_pad.set(1, 1);
	conv1.n_stride.set(1, 1);
	conv1.n_weights.print();
	conv1.n_use_gpu = true;

	pool1.n_pool.set(2, 2);
	pool1.n_stride.set(2, 2);

	relu1.leak = 0;

	//upsc1 needs no parameters

	conv2.set_filter_NCHW(2, 1, 3, 3);
	conv2.n_pad.set(1, 1);
	conv2.n_stride.set(1, 1);
	conv2.n_weights.print();
	conv2.n_use_gpu = true;

	error.n_rsp = data1.n_rsp;
	error.n_use_gpu = false;

	// run gradient descent process
	for (int i = 0; i < 128; i++) {
		conv1.n_use_gpu = false;
		conv1.forward_pass();
		pool1.forward_pass();
		relu1.forward_pass();
		upsc1.forward_pass();
		conv2.n_use_gpu = true;
		conv2.forward_pass();

		error.backward_pass();
		conv2.n_use_gpu = true;
		conv2.backward_pass();
		upsc1.backward_pass();
		relu1.backward_pass();
		pool1.backward_pass();
		conv1.n_use_gpu = false;
		conv1.backward_pass();
		cout << endl;
	}
}

void cnn_example_using_linear_cnn() {
}