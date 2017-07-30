#include "linear_cnn.h"



linear_cnn::linear_cnn()
{
	n_save_folder		= "./";
	n_history_file_name = "history.txt";
	n_current_cnn_name = "current_cnn.txt";
	n_min_cnn_name = "min_cnn.txt";
	n_history = "epoch training_eror validation_error \n";

	n_rng_seed = std::mt19937(0);
	n_d = 0.01f;
	n_batch_size = 2;
	n_max_epoch = 8;
	n_use_gpu = true;
	n_error = DBL_MAX;
	n_epoch_idx = 0;

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