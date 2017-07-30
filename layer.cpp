#include "layer.h"


layer::layer(void)
{
	n_name = "layer";

	n_d = 0.01f;
	leak = 0;
	n_stride.h = 1;
	n_stride.w = 1;
	n_pad.h = 0;
	n_pad.w = 0;
	n_pool.set(2,2);
	n_use_gpu = true;

	n_in1  = NULL; 
	n_out1 = NULL;
}


layer::~layer(void)
{
	int sotp = 1;
}

void layer::set_toy_rsp( int size_c, int size_h, int size_w, float value) {
	n_rsp.resize(1, size_c, size_h, size_w);
	for (int c = 0; c < n_rsp.c() ; c++) {
		for (int h = 0; h < n_rsp.h() ; h++) {
			for (int w = 0; w < n_rsp.w() ; w++) {
				n_rsp(0,c, h, w) = value;
			}
		}
	}
}

//void layer::set_rsp(vector<cv::Mat> images, float min, float max) {
//	int n = images.size();
//	int c = images[0].channels();
//	int h = images[0].rows;
//	int w = images[0].cols;
//	n_rsp.resize(n, c, h, w);
//	for (int nidx = 0; nidx < n_rsp.n(); nidx++ ) {
//		n_rsp.set(images[nidx], nidx, min, max);
//	}
//}
//void layer::set_rsp(vector<cv::Mat> images ) {
//	int n = images.size();
//	int c = images[0].channels();
//	int h = images[0].rows;
//	int w = images[0].cols;
//	n_rsp.resize(n, c, h, w);
//	for (int nidx = 0; nidx < n_rsp.n(); nidx++) {
//		n_rsp.set(images[nidx], nidx);
//	}
//}

void layer::set_rsp(vector<float> labels, int c, int h, int w) {
	int n = labels.size();
	n_rsp.resize( n, c, h, w);
	for (int nidx = 0; nidx < n_rsp.n(); nidx++ ) {
		n_rsp.set(labels[nidx], nidx);
	}
}
//void layer::show_rsp(string window_name, int db, int channel, float min, float max) {
//	cv::Mat image( n_rsp.h(), n_rsp.w(), CV_8U);
//	for (int r = 0; r < n_rsp.h(); r++) {
//		for (int c = 0; c < n_rsp.w(); c++) {
//			float v = n_rsp(db, channel, r, c); 
//			v = 255 * ((v - min) / (max - min));
//			if (v < 0) { v = 0; }
//			if (v > 255) { v = 255; }
//			image.at<uchar>(r, c) = int(v);
//		}
//	}
//	cv::imshow(window_name, image);
//}


