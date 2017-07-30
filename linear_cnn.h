#pragma once
#include <stdio.h>
#include "layer.h"
#include "conv.h"
#include "pool.h"
#include "relu.h"
#include "sqr_error.h"
#include "c4to2x2.h"




//class mylink {
//private:
//	int size;
//public:
//
//	layer *ptr;
//	layer *
//};

class linear_cnn
{
private:

public:
	std::mt19937 n_rng_seed;
	vector<layer*> n_layers;

	float n_d;
	int n_batch_size;
	int n_max_epoch;
	bool  n_use_gpu;
	int n_epoch_idx;
	double n_error;
	string n_save_folder;
	string n_history_file_name;
	string n_current_cnn_name;
	string n_min_cnn_name;
	string n_history;

	linear_cnn();
	~linear_cnn();

	bool save_cnn(string path = "" , int epoch = 0, double error = DBL_MAX );
	bool load_cnn(string path = "" );
	void set_random_weight_init( float min = 0.001f, float max = 0.001f) {
		std::uniform_real_distribution<float> uniform_dist(min, max);
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->set_filter_NCHW(min, max, n_rng_seed) ;
		}
	}
	void set_links() {
		for (int i = 1; i < n_layers.size(); i++ ) {
			n_layers[i]->setinput(n_layers[i - 1]);
		}
	}
	void set_n_d( float d = 0.01f) {
		n_d = d;
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->n_d = n_d;
		}
	}
	void use_gpu( bool ugpu = true ) {
		n_use_gpu = ugpu;
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->n_use_gpu = n_use_gpu;
		}
	}
	double forward_pass() {
		set_links();
		set_n_d(n_d);
		use_gpu(n_use_gpu);
		double avg_error;
		for (int i = 0; i < n_layers.size(); i++ ) {
			avg_error = n_layers[i]->forward_pass();
		}
		return avg_error;
	}
	double backward_pass() {
		for (int i = n_layers.size() - 2; i >= 0; i--) {
			n_layers[i]->backward_pass();
		}
		return 0;
	}
	double optimize(int iteration = 1) {
		double avg_error = 0;
		for (int i = 0; i < iteration; i++) {
			avg_error = forward_pass();
			backward_pass();
		}
		return avg_error;
	}
	inline layer & operator[]( int i) { return *n_layers[i]; }
	inline layer & front() { return *n_layers.front(); }
	inline layer & back() { return *n_layers.back(); }

	void push_back(layer &newLayer) { layer *temp = new layer; *temp = newLayer; n_layers.push_back(temp); }
	void push_back(conv  &newLayer) { conv  *temp = new conv ; *temp = newLayer; n_layers.push_back(temp); }
	void push_back(relu  &newLayer) { relu  *temp = new relu ; *temp = newLayer; n_layers.push_back(temp); }
	void push_back(pool  &newLayer) { pool  *temp = new pool ; *temp = newLayer; n_layers.push_back(temp); }
	void push_back(c4to2x2 &newLayer) { c4to2x2 *temp = new c4to2x2; *temp = newLayer; n_layers.push_back(temp); }
	void push_back(sqr_error  &newLayer) { sqr_error  *temp = new sqr_error; *temp = newLayer;  n_layers.push_back(temp); }
	void pop_back() { 
		if (n_layers.size() > 0) {
			delete n_layers.back();
			n_layers.pop_back();
		}
	}
	void clear() {
		while (n_layers.size() > 0 ) {
			pop_back();
		}
	}
	void print( bool print_all_values = false ) {
		cout << "//////////////////////////////////////////////" << endl;
		cout << "printing linear cnn " << endl;
		cout << "////" << endl;
		cout << "epoch " << n_epoch_idx << " error " << std::scientific << n_error << endl;
		for (int i = 0; i < n_layers.size(); i++ ) {
			cout << "//" << endl;
			n_layers[i]->print(print_all_values);
		}
		cout << "//////////////////////////////////////////////" << endl;
	}
	void nothing_augment(float4d &, float4d &, std::mt19937 ) {}
	void optimize(vector<float4d> &train_imgs, vector<float4d> &train_gts, 
		          vector<float4d> &valid_imgs, vector<float4d> &valid_gts, 
		          void augumentation(float4d &, float4d &, std::mt19937) ) {
		
		double min_val_error = DBL_MAX; // minimum validation error
		int min_epoch = 0;
		load_cnn( n_save_folder + n_min_cnn_name);
		min_val_error = n_error;
		min_epoch = n_epoch_idx;
		load_cnn(n_save_folder + n_current_cnn_name);
		print(); cout << endl;
		
		int training_size = train_imgs.size();
		int validation_size = valid_imgs.size();
		vector<int> train_idx;
		train_idx.clear();
		for (int i = 0; i < training_size; i++) { train_idx.push_back(i); }
		cout << "max epoch is " << n_max_epoch << endl << endl;
		while ( n_epoch_idx < n_max_epoch) {
			n_epoch_idx++;
			std::random_shuffle(train_idx.begin(), train_idx.end());

			cout << n_epoch_idx << " epoch, optimizing training set" << endl;
			double sum_avg_error = 0;
			for (int b = 0; b < training_size; b += n_batch_size) {
				int nsize = n_batch_size;
				if (b + n_batch_size > training_size) {
					nsize = training_size - b;
				}
				n_layers.front()->n_rsp.resize( nsize, train_imgs[0].c(), train_imgs[0].h(), train_imgs[0].w() );
				n_layers.back()->n_rsp.resize(nsize, train_gts[0].c(), train_gts[0].h(), train_gts[0].w());
				for (int i = 0; i < nsize ; i++) {
					int idx = train_idx[b+i];
					float4d temp_img = train_imgs[idx];
					float4d temp_gts = train_gts[ idx];
					augumentation(temp_img, temp_gts, n_rng_seed);
					for (int p = 0; p < temp_img.chw(); p++ ) {
						n_layers.front()->n_rsp(i, p) = temp_img(0, p);
					}
					for (int p = 0; p < temp_gts.chw(); p++) {
						n_layers.back()->n_rsp(i, p) = temp_gts(0, p);
					}
				}
				sum_avg_error += optimize()*nsize;
			}
			double avg_train_error = sum_avg_error / training_size ;
			cout << endl;
			cout << avg_train_error << " average training error during optimization." << endl;
			cout << "calculating error from validation set " << endl;

			////// do not run the pass for training set
			sum_avg_error = 0;
			for (int b = 0; b < validation_size; b += n_batch_size) {
				int nsize = n_batch_size;
				if (b + n_batch_size > validation_size) {
					nsize = validation_size - b;
				}
				n_layers.front()->n_rsp.resize(nsize, valid_imgs[0].c(), valid_imgs[0].h(), valid_imgs[0].w());
				n_layers.back()->n_rsp.resize(nsize, valid_gts[0].c(), valid_gts[0].h(), valid_gts[0].w());
				for (int i = 0; i < nsize ; i++) {
					int idx = train_idx[b + i];
					float4d temp_img = train_imgs[idx];
					float4d temp_gts = train_gts[idx];
					for (int p = 0; p < temp_img.chw(); p++) {
						n_layers.front()->n_rsp(i, p) = temp_img(0, p);
					}
					for (int p = 0; p < temp_gts.chw(); p++) {
						n_layers.back()->n_rsp(i, p) = temp_gts(0, p);
					}
				}
				sum_avg_error += forward_pass()*nsize;
			}
			double avg_verror = sum_avg_error / validation_size;
			save_cnn(n_save_folder + n_current_cnn_name, n_epoch_idx, avg_verror);
			if (min_val_error > avg_verror) {
				min_val_error = avg_verror;
				min_epoch = n_epoch_idx;
				save_cnn(n_save_folder + n_min_cnn_name, n_epoch_idx, avg_verror);
			}
			cout << endl;
			cout << avg_verror << " validation error (avg)." << endl;
			cout << min_val_error << " min validation error (avg) at epoch " << min_epoch << endl << endl;

			//// update history and save history
			n_history += std::to_string(n_epoch_idx);
			n_history += " ";
			n_history += std::to_string(avg_train_error);
			n_history += " ";
			n_history += std::to_string(avg_verror);
			n_history += "\n";
			string history_path = n_save_folder + n_history_file_name;
			std::ofstream outfile(history_path);
			outfile << n_history;
			outfile.close();
		}
	}
};

