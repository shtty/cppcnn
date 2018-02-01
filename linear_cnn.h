////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <stdio.h>
#include "layer.h"
#include "conv.h"
#include "pool.h"
#include "relu.h"
#include "sqr_error.h"
#include "sqr_error_010.h"
#include "c4to2x2.h"
#include "fully.h"
#include "cross_entrophy.h"
#include "residual.h"


class linear_cnn
{
private:
	
public:
	vector<layer*> n_layers;

	float4d n_rsp;
	float n_d;
	float n_validation_proportion;
	int n_batch_size;
	int n_max_epoch;
	bool  n_use_gpu;
	int n_epoch_idx;
	int n_min_epoch;
	double n_error;
	double n_min_val_error;
	string n_save_folder;
	string n_history_file_name;
	string n_current_cnn_name;
	string n_min_cnn_name;
	string n_history;

	void (*data_augument_function)(float4d &, float4d & );
	void (*show_progress_function)(vector<layer*> &);

	linear_cnn();
	~linear_cnn();

	void optimize(vector<float4d> &db_imgs, vector<float4d> &db_gts,
		vector<int> &train_idx, vector<int> &valid_idx, bool load_from_previous = false);
	void optimize( vector<float4d> &training_imgs, vector<float4d> &training_gts, bool load_from_previous = false );
	bool save_cnn(string path = "" , int epoch = 0, double error = DBL_MAX );
	bool load_cnn(string path = "" );
	void print(bool print_all_values = false);

	void forward_pass_test() {
		set_links();
		set_n_d(n_d);
		use_gpu(n_use_gpu);
		double avg_error = 0;
		for (int i = 0; i < n_layers.size()-1; i++) {
			avg_error = n_layers[i]->forward_pass();
		}
		//// the last layer is the error
		n_rsp = n_layers[n_layers.size() - 2]->n_rsp;
	}
	double forward_pass( ) {
		set_links();
		set_n_d(n_d);
		use_gpu(n_use_gpu);
		double avg_error = 0;
		for (int i = 0; i < n_layers.size(); i++ ) {
			avg_error = n_layers[i]->forward_pass();
		}
		//// the last layer is the error
		n_rsp = n_layers[n_layers.size() - 2]->n_rsp;
		return avg_error;
	}
	double backward_pass( bool update_weights = true ) {
		for (int i = n_layers.size() - 1; i >= 0; i--) {
			n_layers[i]->backward_pass(update_weights);
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
	void	filter_init(string init_method = "xavier") {
		for (int i = 0; i < n_layers.size(); i++) {
			n_layers[i]->n_weights_set(init_method);
		}
	}
	void	set_links() {
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->set_input(*n_layers[i - 1]);
		}
	}
	void	set_n_d(float d = 0.01f) {
		n_d = d;
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->set_gradient_step_size(n_d);
		}
	}
	void	use_gpu(bool ugpu = true) {
		n_use_gpu = ugpu;
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->n_use_gpu = n_use_gpu;
		}
	}
	void	set_filters(string init_method) {
		for (int i = 1; i < n_layers.size(); i++) {
			n_layers[i]->n_weights_set(init_method);
		}
	}

	inline layer & operator[]( int i) { return *n_layers[i]; }
	inline layer & operator()(int i) { return *n_layers[i]; }
	inline layer & at(int i) { return *n_layers[i]; }
	inline layer & front() { return *n_layers.front(); }
	inline layer & back() { return *n_layers.back(); }
	inline int size() { return n_layers.size(); }
	template<class T>
	void push_back(T &newlayer) { T *temp = new T; *temp = newlayer; n_layers.push_back(temp); }
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
	
};

