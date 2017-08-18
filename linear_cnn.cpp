#include "linear_cnn.h"



linear_cnn::linear_cnn()
{
	n_save_folder		= "./";
	n_history_file_name = "history.txt";
	n_current_cnn_name = "current_cnn.txt";
	n_min_cnn_name = "min_cnn.txt";
	n_history = "epoch training_eror validation_error \n";

	
	n_d = 0.01f;
	n_batch_size = 2;
	n_max_epoch = 8;
	n_use_gpu = true;
	n_error = DBL_MAX;
	n_epoch_idx = 0;

	data_augument_function = NULL;
	show_progress_function = NULL;

}


linear_cnn::~linear_cnn()
{
	clear();
}

bool linear_cnn::save_cnn(string path, int epoch, double error) {
	if ( path == "" ) {
		path = n_save_folder + n_current_cnn_name;
	}
	std::ofstream myfile(path);
	if (myfile.is_open()) {
		myfile << "epoch " << epoch << " error " << error << endl;
		myfile << endl << n_layers.size() << endl << endl;
		for (int k = 0; k < n_layers.size(); k++) {
			n_layers[k]->save_init(myfile);
		}
		for (int k = 0; k < n_layers.size(); k++) {
			n_layers[k]->save_weights(myfile);
		}
		myfile.close();

		return true;
	}
	return false;
}

bool linear_cnn::load_cnn(string path) {
	if (path == "") {
		path = n_save_folder + n_current_cnn_name;
	}
	bool load_success = false;
	ifstream myfile(path);
	if (myfile.is_open()) {
		clear();
		///// load history //////////////////////////////////////
		ifstream hist_file(n_save_folder + n_history_file_name);
		if (hist_file.is_open()) {
			n_history = "";
			string line;
			while (getline(hist_file, line)) {
				n_history += line + "\n";
			}
		}
		hist_file.close();
		//////////////////////////////////////////////////////////

		string epcherrorstr;
		float temp_error;
		int layer_size;
		myfile >> epcherrorstr;
		myfile >> n_epoch_idx;
		myfile >> epcherrorstr;
		myfile >> n_error;
		myfile >> layer_size;
		
		for (int k = 0; k < layer_size; k++) {
			string type, strtemp;;
			myfile >> type;
			if (!type.compare("conv")) {
				conv temp_conv;
				temp_conv.load_init(myfile, type);
				push_back(temp_conv);
			}
			else if (!type.compare("relu")) {
				relu temp_relu;
				temp_relu.load_init(myfile, type);
				push_back(temp_relu);
			}
			else if (!type.compare("pool")) {
				pool temp_pool;
				temp_pool.load_init(myfile, type);
				push_back(temp_pool);
			}
			else if (!type.compare("sqr_error")) {
				sqr_error temp_error;
				temp_error.load_init(myfile, type);
				push_back(temp_error);
			}
			else if (!type.compare("sqr_error_010")) {
				sqr_error_010 temp_error;
				temp_error.load_init(myfile, type);
				push_back(temp_error);
			}
			else if (!type.compare("c4to2x2")) {
				c4to2x2 temp_c4to1;
				temp_c4to1.load_init(myfile, type);
				push_back(temp_c4to1);
			}
			else { // "layer"
				layer temp_layer;
				temp_layer.load_init(myfile, type);
				push_back(temp_layer);
			}
		}

		for (int k = 0; k < layer_size; k++) {
			n_layers[k]->load_weights(myfile);
		}
		myfile.close();
		load_success = true;
	}
	else {
		cout << endl << "FAILED to LOAD CNN" << endl;
	}
	return load_success;
}
void linear_cnn::print(bool print_all_values) {
	cout << "//////////////////////////////////////////////" << endl;
	cout << "printing linear cnn " << endl;
	cout << "////" << endl;
	cout << "epoch " << n_epoch_idx << " error " << std::scientific << n_error << endl;
	for (int i = 0; i < n_layers.size(); i++) {
		cout << "//" << endl;
		n_layers[i]->print(print_all_values);
	}
	cout << "//////////////////////////////////////////////" << endl;
}

void linear_cnn::optimize(vector<float4d> &db_imgs, vector<float4d> &db_gts,
	                      vector<int> &train_idx, vector<int> &valid_idx, bool load_from_previous ) {

	double min_val_error = DBL_MAX; // minimum validation error
	int min_epoch = 0;
	if (load_from_previous) {
		load_cnn(n_save_folder + n_min_cnn_name);
		min_val_error = n_error;
		min_epoch = n_epoch_idx;
		load_cnn(n_save_folder + n_current_cnn_name);
		print(); cout << endl;
	}

	int training_size = train_idx.size();
	int validation_size = valid_idx.size();

	cout << "max epoch is " << n_max_epoch << endl << endl;
	while (n_epoch_idx < n_max_epoch) {
		n_epoch_idx++;
		std::random_shuffle(train_idx.begin(), train_idx.end());

		cout << n_epoch_idx << " epoch, optimizing training set" << endl;
		double sum_avg_error = 0;
		for (int b = 0; b < training_size; b += n_batch_size) {

			int nsize = n_batch_size;
			if (b + n_batch_size > training_size) {
				nsize = training_size - b;
			}
			n_layers.front()->n_rsp.resize(nsize, db_imgs[0].c(), db_imgs[0].h(), db_imgs[0].w());
			n_layers.back()->n_rsp.resize(nsize, db_gts[0].c(), db_gts[0].h(), db_gts[0].w());
			for (int i = 0; i < nsize; i++) {
				int idx = train_idx[b + i];
				float4d temp_img = db_imgs[idx];
				float4d temp_gts = db_gts[idx];

				if (data_augument_function != NULL) {
					data_augument_function(temp_img, temp_gts);
				}
				for (int p = 0; p < temp_img.chw(); p++) {
					n_layers.front()->n_rsp(i, p) = temp_img(0, p);
				}
				for (int p = 0; p < temp_gts.chw(); p++) {
					n_layers.back()->n_rsp(i, p) = temp_gts(0, p);
				}
			}
			sum_avg_error += optimize()*nsize;
			cout << "  " << b << "/" << training_size << " ";
		}
		if (show_progress_function != NULL) {
			show_progress_function(n_layers);
		};


		double avg_train_error = sum_avg_error / training_size;
		cout << endl;
		cout << avg_train_error << " average training error during optimization." << endl;
		cout << "calculating error from validation set " << endl;

		////// 
		double avg_verror = 0;
		if (validation_size > 0) {
			sum_avg_error = 0;
			for (int b = 0; b < validation_size; b += n_batch_size) {
				int nsize = n_batch_size;
				if (b + n_batch_size > validation_size) {
					nsize = validation_size - b;
				}
				n_layers.front()->n_rsp.resize(nsize, db_imgs[0].c(), db_imgs[0].h(), db_imgs[0].w());
				n_layers.back()->n_rsp.resize(nsize, db_gts[0].c(), db_gts[0].h(), db_gts[0].w());
				for (int i = 0; i < nsize; i++) {
					int idx = valid_idx[b + i];
					float4d temp_img = db_imgs[idx];
					float4d temp_gts = db_gts[idx];
					for (int p = 0; p < temp_img.chw(); p++) {
						n_layers.front()->n_rsp(i, p) = temp_img(0, p);
					}
					for (int p = 0; p < temp_gts.chw(); p++) {
						n_layers.back()->n_rsp(i, p) = temp_gts(0, p);
					}
				}
				sum_avg_error += forward_pass()*nsize;
			}
			avg_verror = sum_avg_error / validation_size;
			if (min_val_error > avg_verror) {
				min_val_error = avg_verror;
				min_epoch = n_epoch_idx;
				save_cnn(n_save_folder + n_min_cnn_name, n_epoch_idx, avg_verror);
			}
			cout << endl;
			cout << avg_verror << " validation error (avg)." << endl;
			cout << min_val_error << " min validation error (avg) at epoch " << min_epoch << endl << endl;
		}
		save_cnn(n_save_folder + n_current_cnn_name, n_epoch_idx, avg_verror);

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