////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <stdio.h>
#include "linear_cnn.h"


void cnn_example_using_layer() {
	//// a simple generative model test
	layer   data1;
	conv    conv1;
	pool    pool1;
	relu    relu1;
	c4to2x2 upsc1;
	conv    conv2;
	relu	relu2;
	conv	conv3;
	sqr_error error;

	//// set in and out of layers
	//// data1->conv1->pool1->relu1->upsc1->conv2->error
	conv1.set_input(data1);
	pool1.set_input(conv1);
	relu1.set_input(pool1);
	upsc1.set_input(relu1);
	conv2.set_input(upsc1);
	relu2.set_input(conv2);
	conv3.set_input(relu2);
	error.set_input(conv3);

	// set parameters on the layers
	// some parameters are default values at the constructor
	data1.n_rsp.resize(2, 2, 4, 4);
	for (int i = 0; i < 64; i++ ) {
		data1.n_rsp(i) = float(i % 3);
	}
	conv1.n_weights_bias_set(16, 2, 3, 3);
	conv1.n_zero_padding = true;

	pool1.n_pool.set(2, 2);
	pool1.n_stride.set(2, 2);

	relu1.n_leak = 0.0f;

	//upsc1 has no parameters
	//resize 4 channel into 1 chanel 2x2

	conv2.n_weights_bias_set(16, 4, 3, 3);
	conv2.n_zero_padding = true;
	conv2.n_stride.set(1, 1);

	relu2.n_leak = 0.0f;

	conv3.n_weights_bias_set(2, 16, 3, 3);
	conv3.n_zero_padding = true;
	conv3.n_stride.set(1, 1);

	error.n_rsp = data1.n_rsp;

	/// Saves the initial CNN
	string path = "cnn_example_using_layer.txt";
	std::ofstream savefile(path);
	data1.save_init(savefile);
	conv1.save_init(savefile);
	pool1.save_init(savefile);
	relu1.save_init(savefile);
	upsc1.save_init(savefile);
	conv2.save_init(savefile);
	relu2.save_init(savefile);
	conv3.save_init(savefile);
	error.save_init(savefile);
	conv1.save_weights(savefile);
	conv2.save_weights(savefile);
	conv3.save_weights(savefile);
	savefile.close();

	/// print initial parameters
	data1.print();
	conv1.print();
	pool1.print();
	relu1.print();
	upsc1.print();
	conv2.print();
	relu2.print();
	conv3.print();
	error.print();


	// run gradient descent process
	cout << endl;
	for (int i = 0; i < 1024; i++) {
		cout << i << " ";
	
		conv1.forward_pass();
		pool1.forward_pass();
		relu1.forward_pass();
		upsc1.forward_pass();
		conv2.forward_pass();
		relu2.forward_pass();
		conv3.forward_pass();
		error.forward_pass();

		error.backward_pass();
		conv3.backward_pass();
		relu2.backward_pass();
		conv2.backward_pass();
		upsc1.backward_pass();
		relu1.backward_pass();
		pool1.backward_pass();
		conv1.backward_pass();
		cout <<  endl;
		
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
	relu2.load_init(loadfile);
	conv3.load_init(loadfile);
	error.load_init(loadfile);
	conv1.load_weights(loadfile);
	conv2.load_weights(loadfile);
	conv3.load_weights(loadfile);
	loadfile.close();

	/// print reloaded initial parameters
	// this should be same as the initial parameters
	data1.print();
	conv1.print();
	pool1.print();
	relu1.print();
	upsc1.print();
	conv2.print();
	relu2.print();
	conv3.print();
	error.print();
}

void cnn_example_using_linear_cnn() {
	linear_cnn cnn;

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
	data_temp.n_rsp.resize(2, 2, 4, 4);
	data_temp.n_rsp.set(0, 0);
	data_temp.n_rsp.set(1, 1);
	cnn.push_back(data_temp);

	conv_temp.n_weights_bias_set(4, 2, 3, 3);
	conv_temp.n_zero_padding = true;
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	pool_temp.n_pool.set(2, 2);
	pool_temp.n_stride.set(2, 2);
	cnn.push_back(pool_temp);

	relu_temp.n_leak = 0;
	cnn.push_back(relu_temp);

	//upsc_temp needs no parameters
	cnn.push_back(upsc_temp);

	conv_temp.n_weights_bias_set(2,1, 3, 3);
	conv_temp.n_zero_padding = true;
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	relu_temp.n_leak = 0;
	cnn.push_back(relu_temp);

	conv_temp.n_weights_bias_set(2, 2, 3, 3);
	conv_temp.n_zero_padding = true;
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	error_temp.n_rsp = data_temp.n_rsp;
	cnn.push_back(error_temp);
	


	cnn.n_save_folder = "./";
	cnn.n_current_cnn_name = "linear_cnn_current.txt";
	cnn.n_min_cnn_name = "linear_cnn_min.txt";
	cnn.n_history_file_name = "linear_cnn_history.txt";

	std::uniform_real_distribution<float> uniform_dist(0, 1);
	std::uniform_real_distribution<float> uniform_dist_valid(0.5f, 0.52f);
	double min_val_error = DBL_MAX; // minimum validation error
	if ( cnn.load_cnn( cnn.n_save_folder + cnn.n_min_cnn_name ) ) {
		min_val_error = cnn.n_error;
	}
	cnn.print(); cout << endl;

	

	int min_epoch = 0;
	int batch_size = 2; // 
	int training_size = 128;
	int validation_size = 32;
	int max_epoch = 128;
	cout << "max epoch is " << max_epoch << endl;
	while( cnn.n_epoch_idx < max_epoch ) {
		cnn.n_epoch_idx++;

		cout << cnn.n_epoch_idx << " epoch, optimizing training set" << endl;
		double sum_avg_error = 0;
		for (int b = 0; b < training_size; b += batch_size) {
			///// get training batch ////
			for (int i = 0; i < batch_size; i++ ) {
				float v = uniform_dist(float4d::n_random_seed);
				while ( v > 0.5f && v < 0.52f ) {
					v = uniform_dist(float4d::n_random_seed);
				}
				cnn.front().n_rsp.set(v, i); // data layer
				cnn.back().n_rsp.set( v, i); // error layer
			}
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
			///// set validation set /////////////
			for (int i = 0; i < batch_size; i++) {
				float v = uniform_dist_valid(float4d::n_random_seed);
				cnn.front().n_rsp.set(v, i); // data layer
				cnn.back().n_rsp.set(v, i); // error layer
			}
			//////////////////////////////////////
			sum_avg_error += cnn.forward_pass();
		}
		double avg_verror = sum_avg_error / (validation_size / batch_size);
		//cnn.save_cnn( cnn.n_save_folder + cnn.n_current_cnn_name, cnn.n_epoch_idx, avg_verror );
		if ( min_val_error > avg_verror ) {
			min_val_error = avg_verror;
			min_epoch = cnn.n_epoch_idx;
			//cnn.save_cnn(cnn.n_save_folder + cnn.n_min_cnn_name, cnn.n_epoch_idx, avg_verror);
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
		//outfile << cnn.n_history;
		outfile.close();
	}

	cnn.load_cnn(cnn.n_save_folder + cnn.n_min_cnn_name);
	cout << endl;
	cnn.print();
}

void data_augument( float4d &image, float4d &ground_truth ) {
	std::uniform_real_distribution<float> uniform_dist(-0.1f, 0.1f);
	for (int i = 0; i < image.nchw(); i++) {
		image[i] += uniform_dist(float4d::n_random_seed);
	}
	// add noise to the input image
	// no changes in gorund truth
}
void cnn_example_using_linear_cnn_simple( bool load_from_previous = true) {
	linear_cnn cnn;

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
	data_temp.n_rsp.resize(2, 2, 4, 4);
	cnn.push_back(data_temp);

	conv_temp.n_weights_bias_set(4, 2, 3, 3);
	conv_temp.n_zero_padding = true;
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	pool_temp.n_pool.set(2, 2);
	pool_temp.n_stride.set(2, 2);
	cnn.push_back(pool_temp);

	relu_temp.n_leak = 0;
	cnn.push_back(relu_temp);

	//upsc_temp needs no parameters
	cnn.push_back(upsc_temp);

	conv_temp.n_weights_bias_set(2, 1, 3, 3);
	conv_temp.n_zero_padding = true;
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	relu_temp.n_leak = 0;
	cnn.push_back(relu_temp);

	conv_temp.n_weights_bias_set(2, 2, 3, 3);
	conv_temp.n_zero_padding = true;
	conv_temp.n_stride.set(1, 1);
	conv_temp.n_use_gpu = true;
	cnn.push_back(conv_temp);

	error_temp.n_rsp = data_temp.n_rsp;
	cnn.push_back(error_temp);

	////////////////////////////////////////////////

	//// Creating Training and Validation Set //////////////////////////////
	std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
	int training_size = 511;
	int validation_size = 128;
	vector<float4d> training_images;
	vector<float4d> training_groundtruths;
	vector<int> db_idx;
	db_idx.clear();
	for (int i = 0; i < training_size; i++ ) {
		float4d f4dtemp;
		f4dtemp.resize(1, 2, 4, 4); // image is 2 channel 8 by 8 image;
		f4dtemp.set(uniform_dist(float4d::n_random_seed));
		training_images.push_back(f4dtemp);
		training_groundtruths.push_back(f4dtemp);
		// since we are making generative network ground truth is same as input images
		db_idx.push_back(i);
	}
	random_shuffle(db_idx.begin(), db_idx.end());
	vector<int> train_idx;
	vector<int> valid_idx;
	train_idx.clear(); valid_idx.clear();
	for (int i = 0; i < validation_size; i++) {
		valid_idx.push_back( db_idx.back() );
		db_idx.pop_back();
	}
	while ( !db_idx.empty() ) {
		train_idx.push_back(db_idx.back());
		db_idx.pop_back();
	}
	//////////////////////////////////////////////////////////////////////
	
	cnn.n_save_folder = "./";
	cnn.n_current_cnn_name = "linear_cnn_simple_current.txt";
	cnn.n_min_cnn_name = "linear_cnn_simple_min.txt";
	cnn.n_history_file_name = "linear_cnn_simple_history.txt";
	cnn.n_d = 0.01f; // set gradient step
	cnn.n_batch_size = 2;
	cnn.n_use_gpu = true;
	cnn.n_max_epoch = 64;
	cnn.data_augument_function = &data_augument;

	cnn.optimize(training_images, training_groundtruths, train_idx, valid_idx, load_from_previous );
	
	cnn.load_cnn(cnn.n_save_folder + cnn.n_min_cnn_name);
	cnn.print();
}