#pragma once
#include <stdio.h>
#include "linear_cnn.h"

void cnn_example_using_layer();
void cnn_example_using_linear_cnn();
void  cnn_example_using_linear_cnn_simple();

void cnn_example_using_layer() {
	//// a simple generative model test
	layer   data1;
	conv    conv1;
	pool    pool1;
	relu    relu1;
	c4to2x2 upsc1;
	conv    conv2;
	sqr_error error;

	//// set in and out of layers
	//// data1->conv1->pool1->relu1->upsc1->conv2->error
	conv1.setinput(&data1);
	pool1.setinput(&conv1);
	relu1.setinput(&pool1);
	upsc1.setinput(&relu1);
	conv2.setinput(&upsc1);
	error.setinput(&conv2);

	// set parameters on the layers
	// some parameters are default values at the constructor
	data1.n_rsp.resize(2, 2, 8, 8);
	data1.n_rsp.set(0.5f);
	
	conv1.set_filter_NCHW(4, 2, 3, 3);
	conv1.n_pad.set(1, 1);
	conv1.n_stride.set(1, 1);

	pool1.n_pool.set(2, 2);
	pool1.n_stride.set(2, 2);

	relu1.leak = 0;

	//upsc1 needs no parameters

	conv2.set_filter_NCHW(2, 1, 3, 3);
	conv2.n_pad.set(1, 1);
	conv2.n_stride.set(1, 1);

	error.n_rsp = data1.n_rsp;
	error.n_use_gpu = false;

	/// Saves the initial CNN
	string path = "cnn_example_using_layer.txt";
	std::ofstream savefile(path);
	data1.save_init(savefile);
	conv1.save_init(savefile);
	pool1.save_init(savefile);
	relu1.save_init(savefile);
	upsc1.save_init(savefile);
	conv2.save_init(savefile);
	error.save_init(savefile);
	conv1.save_weights(savefile);
	conv2.save_weights(savefile);
	savefile.close();

	/// print initial parameters
	data1.print(true);
	conv1.print();
	pool1.print();
	relu1.print();
	upsc1.print();
	conv2.print();
	error.print();


	// run gradient descent process
	cout << endl;
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
	cout << endl;

	
	// reload the initial CNN
	std::ifstream loadfile(path);
	data1.load_init(loadfile);
	conv1.load_init(loadfile);
	pool1.load_init(loadfile);
	relu1.load_init(loadfile);
	upsc1.load_init(loadfile);
	conv2.load_init(loadfile);
	error.load_init(loadfile);
	conv1.load_weights(loadfile);
	conv2.load_weights(loadfile);
	loadfile.close();

	/// print reloaded initial parameters
	// this should be same as the initial parameters
	data1.print(true);
	conv1.print();
	pool1.print();
	relu1.print();
	upsc1.print();
	conv2.print();
	error.print();

}

void cnn_example_using_linear_cnn() {
	std::mt19937 rng(0); // random number seed for all initializing and stochastic gradients
	linear_cnn cnn;
	cnn.n_rng_seed = rng;

	//// a simple generative model test
	layer data_temp;
	conv  conv_temp;
	pool  pool_temp;
	relu  relu_temp;
	c4to2x2 upsc_temp;
	sqr_error error_temp;

	cnn.clear();
	// set parameters on the layers
	// some parameters are default values at the constructor
	data_temp.n_rsp.resize(2, 2, 8, 8);
	data_temp.n_rsp.set(0.5f);
	cnn.push_back(data_temp);

	conv_temp.set_filter_NCHW(4, 2, 3, 3, rng);
	conv_temp.n_pad.set(1, 1);
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	pool_temp.n_pool.set(2, 2);
	pool_temp.n_stride.set(2, 2);
	cnn.push_back(pool_temp);

	relu_temp.leak = 0;
	cnn.push_back(relu_temp);

	//upsc_temp needs no parameters
	cnn.push_back(upsc_temp);

	conv_temp.set_filter_NCHW(2, 1, 3, 3, rng);
	conv_temp.n_pad.set(1, 1);
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	error_temp.n_rsp = data_temp.n_rsp;
	cnn.push_back(error_temp);
	
	cnn.n_save_folder = "./";
	cnn.n_current_cnn_name = "linear_cnn_current.txt";
	cnn.n_min_cnn_name = "linear_cnn_min";
	cnn.n_history_file_name = "linear_cnn_history";

	std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
	double min_val_error = DBL_MAX; // minimum validation error
	if ( cnn.load_cnn( cnn.n_save_folder + cnn.n_min_cnn_name ) ) {
		min_val_error = cnn.n_error;
	}
	cnn.load_cnn(cnn.n_save_folder + cnn.n_current_cnn_name );
	cnn.print(); cout << endl;

	int min_epoch = 0;
	int batch_size = 2; // 
	int training_size = 512;
	int validation_size = 256;
	int max_epoch = 32;
	cout << "max epoch is " << max_epoch << endl;
	while( cnn.n_epoch_idx < max_epoch ) {
		cnn.n_epoch_idx++;
		
		cout << cnn.n_epoch_idx << " epoch, optimizing training set" << endl;
		double sum_avg_error = 0;
		for (int b = 0; b < training_size; b += batch_size) {
			///// get training batch ////
			float v1, v2;
			v1 = uniform_dist(rng);
			v2 = uniform_dist(rng);
			cnn[0].n_rsp.set(v1, 0); // data layer
			cnn[0].n_rsp.set(v2, 1);
			cnn[6].n_rsp.set(v1, 0); // error layer
			cnn[6].n_rsp.set(v2, 1);
			//////////////////////////////
			sum_avg_error += cnn.optimize();
		}
		double avg_train_error = sum_avg_error / (training_size / batch_size);
		cout << endl;
		cout << avg_train_error << " average training error during optimization." << endl;
		cout << "calculating error from validation set " << endl;

		////// do not run the pass for training set
		sum_avg_error = 0;
		for (int b = 0; b < validation_size; b += batch_size) {
			float v1, v2;
			v1 = uniform_dist(rng);
			v2 = uniform_dist(rng);
			cnn.front().n_rsp.set(v1, 0);
			cnn.front().n_rsp.set(v2, 1);
			cnn.back().n_rsp.set(v1, 0);
			cnn.back().n_rsp.set(v2, 1);
			/////////////////////
			sum_avg_error += cnn.forward_pass();
			
		}
		double avg_verror = sum_avg_error / (validation_size / batch_size);
		cnn.save_cnn( cnn.n_save_folder + cnn.n_current_cnn_name, cnn.n_epoch_idx, avg_verror );
		if ( min_val_error > avg_verror ) {
			min_val_error = avg_verror;
			min_epoch = cnn.n_epoch_idx;
			cnn.save_cnn(cnn.n_save_folder + cnn.n_min_cnn_name, cnn.n_epoch_idx, avg_verror);
		}
		cout << endl;
		cout << avg_verror << " validation error (avg)." << endl;
		cout << min_val_error << " min validation error (avg) at epoch " << min_epoch << endl << endl;

		// update history and save history
		cnn.n_history += std::to_string(cnn.n_epoch_idx);
		cnn.n_history += " ";
		cnn.n_history += std::to_string(avg_train_error);
		cnn.n_history += " ";
		cnn.n_history += std::to_string(avg_verror);
		cnn.n_history += "\n";
		string history_path = cnn.n_save_folder + cnn.n_history_file_name;
		std::ofstream outfile(history_path);
		outfile << cnn.n_history;
		outfile.close();
	}

	cnn.load_cnn(cnn.n_save_folder + cnn.n_min_cnn_name);
	cout << endl;
	cnn.print();
}

void data_augument( float4d &image, float4d &ground_truth, std::mt19937 rng  ) {
	std::uniform_real_distribution<float> uniform_dist(-0.1f, 0.1f);
	for (int i = 0; i < image.nchw(); i++) {
		image[i] += uniform_dist(rng);
	}
	// add noise to the input image
	// no changes in gorund truth
}
void cnn_example_using_linear_cnn_simple() {
	std::mt19937 rng(0); // random number seed for all initializing and stochastic gradients
	linear_cnn cnn;
	cnn.n_rng_seed = rng;

	//// a simple generative model test
	layer data_temp;
	conv  conv_temp;
	pool  pool_temp;
	relu  relu_temp;
	c4to2x2 upsc_temp;
	sqr_error error_temp;

	cnn.clear();
	// set parameters on the layers
	// some parameters are default values at the constructor
	data_temp.n_rsp.resize(2, 2, 8, 8);
	data_temp.n_rsp.set(0.5f);
	cnn.push_back(data_temp);

	conv_temp.set_filter_NCHW(4, 2, 3, 3);
	conv_temp.n_pad.set(1, 1);
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	pool_temp.n_pool.set(2, 2);
	pool_temp.n_stride.set(2, 2);
	cnn.push_back(pool_temp);

	relu_temp.leak = 0;
	cnn.push_back(relu_temp);

	//upsc_temp needs no parameters
	cnn.push_back(upsc_temp);

	conv_temp.set_filter_NCHW(2, 1, 3, 3);
	conv_temp.n_pad.set(1, 1);
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	error_temp.n_rsp = data_temp.n_rsp;
	cnn.push_back(error_temp);
	////////////////////////////////////////////////

	//// Creating Training and Validation Set //////////////////////////////
	std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
	int training_size = 511;
	int validation_size = 255;
	vector<float4d> training_images;
	vector<float4d> training_groundtruths;
	vector<float4d> validation_images;
	vector<float4d> validation_groundtruths;
	for (int i = 0; i < training_size; i++ ) {
		float4d f4dtemp;
		f4dtemp.resize(1, 2, 8, 8); // image is 2 channel 8 by 8 image;
		f4dtemp.set(uniform_dist(rng));
		training_images.push_back(f4dtemp);
		training_groundtruths.push_back(f4dtemp); 
		// since we are making generative network ground truth is same as input images
	}
	for (int i = 0; i < validation_size; i++) {
		float4d f4dtemp;
		f4dtemp.resize(1, 2, 8, 8);
		f4dtemp.set(uniform_dist(rng));
		validation_images.push_back(f4dtemp);
		validation_groundtruths.push_back(f4dtemp);
	}
	//////////////////////////////////////////////////////////////////////
	
	cnn.n_save_folder = "./";
	cnn.n_current_cnn_name = "linear_cnn_simple_current.txt";
	cnn.n_min_cnn_name = "linear_cnn_simple_min.txt";
	cnn.n_history_file_name = "linear_cnn_simple_history.txt";
	cnn.n_d = 0.01f; // set gradient step
	cnn.n_batch_size = 7;
	cnn.n_use_gpu = true;
	cnn.set_random_weight_init(0.001f,0.001f);
	cnn.n_max_epoch = 16;

	cnn.optimize(training_images, training_groundtruths, validation_images, validation_groundtruths, data_augument);
	
	cnn.load_cnn(cnn.n_save_folder + cnn.n_min_cnn_name);
	cnn.print();
}