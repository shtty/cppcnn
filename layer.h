#pragma once
#define SHTTY_CUDNN

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
using namespace std;
//#include <opencv/cv.hpp>
//#include <opencv/highgui.h>



class size4d {
public:
	int n, c, h, w;
	size4d(int in = 0, int ic = 0, int ih = 0, int iw = 0) { set(in,ic,ih,iw); }
	inline void set(int in, int ic, int ih, int iw) {
		n = in; c = ic;  h = ih; w = iw;
	}
	inline int nchw() { return  n*c*h*w; }
	inline bool operator!=( size4d in ) {
		if ( n == in.n && c == in.c && h == in.h && w == in.w) { return true; }
		return false;
	}
};
class size2d {
public:
	int w, h;
	size2d() { w = h = 0; }
	inline void set( int ih, int iw) { h = ih; w = iw; }
};
class float1d {
private:
	int n_size;
	vector<float> p_v;
public:
	float1d() {}
	~float1d() {}
	inline void resize(int new_size) { 
		if ( new_size != n_size) {
			p_v.resize(new_size);
		}
		n_size = new_size;
	}
	inline void set( float v ) { for (int p = 0; p < n_size; p++) { p_v[p] = v;  } }
	inline int size() { return n_size; }
	inline float & at( int n ) { return p_v[n]; }
	inline float & operator()(int n) { return p_v[n]; }
	void print( bool print_values = false ) {
		cout << "size1d " << n_size << ": ";
		if (print_values) {
			for (int i = 0; i < n_size; i++) { cout << at(i) << " "; }
		}
		cout << endl;
	}
};
class float4d { // N C H W
private:
	size4d n_size;
	vector<float> p_v;
	int xw, xwxh, xwxhxc, xwxhxcxn;
	float n_zero;
public:
	

	float4d(int n = 0, int c = 0, int h = 0, int w = 0) { 
		resize(n,c,h,w); 
	}
	float4d(size4d s) { 
		resize(s); 
	}
	inline void resize(int n, int c, int h, int w) { 
		n_size.set( n, c, h, w);
		resize(n_size);
	}
	inline void resize(size4d insize ) {
		n_size = insize;
		if( p_v.size() != n_size.nchw() ) {
			p_v.resize(n_size.nchw());
		}
		xw       =        n_size.w;
		xwxh     = xw    *n_size.h;
		xwxhxc   = xwxh  *n_size.c;
		xwxhxcxn = xwxhxc*n_size.n;
	}
	
	inline int nchw2idx(int n, int c, int h, int w) { return n*xwxhxc + c*xwxh + h*xw + w; }
	inline float & at(int n, int c, int h, int w) {
		if (h < 0 || w < 0 || h >= n_size.h || w >= n_size.w) {
			n_zero = 0;
			return n_zero; //cout << "returninging 0s ";
		}
		else {
			return p_v[nchw2idx(n, c, h, w)];
		}
	}
	inline float & at(int p) { return p_v[p]; }
	inline float & operator()(int p) { return p_v[p]; }
	inline float & operator[](int p) { return p_v[p]; }
	inline float & operator()(int n, int c, int h, int w) { return p_v[nchw2idx(n, c, h, w)]; }
	inline float & operator()(int n, int p ) { return p_v[n*xwxhxc + p]; }

	inline size4d size() { return n_size; }
	inline int nchw() { return xwxhxcxn; }
	inline int  chw() { return xwxhxc  ; }
	inline int w() { return n_size.w;  }
	inline int h() { return n_size.h; }
	inline int c() { return n_size.c; }
	inline int n() { return n_size.n; }
	
	inline void set(float value, int nidx) {
		std::fill(p_v.begin() + nidx*xwxhxc, p_v.begin() + nidx*xwxhxc + xwxhxc - 1, value);
	}
	inline void set(float value) { 
		std::fill(p_v.begin(), p_v.end(), value); 
	}
	void set(float min, float max, std::mt19937 rng = std::mt19937(0) ) {
		std::uniform_real_distribution<float> uniform_dist(min, max);
		for (int p = 0; p < xwxhxcxn; p++) {
			p_v[p] = uniform_dist(rng);
		}
	}
	//void set(cv::Mat image, int sidx, float min, float max) {
	//	for (int r = 0; r < n_size.h; r++) {
	//		for (int c = 0; c < n_size.w; c++) {
	//			for (int ch = 0; ch < n_size.c; ch++) {
	//				float v = float(image.data[r*n_size.w * n_size.c + c*n_size.c + ch]);
	//				at(sidx, ch, r, c) = (v / 255)*(max-min) + min;
	//			}
	//		}
	//	}
	//}
	//void set(cv::Mat image, int sidx ) {
	//	for (int r = 0; r < n_size.h; r++) {
	//		for (int c = 0; c < n_size.w; c++) {
	//			for (int ch = 0; ch < n_size.c; ch++) {
	//				float v = float(image.data[r*n_size.w * n_size.c + c*n_size.c + ch]);
	//				at(sidx, ch, r, c) = v;
	//			}
	//		}
	//	}
	//}
	void print(bool print_values = false) {
		cout << "size_NCHW " << n_size.n << " " << n_size.c << " " << n_size.h << " " << n_size.w << endl;
		if (print_values) {
			for (int ps = 0; ps < n_size.n; ps++) { // per each image
				for (int pc = 0; pc < n_size.c; pc++) { // per each channel
					for (int py = 0; py < n_size.h; py++) { // per each y 
						for (int px = 0; px < n_size.w; px++) { // per each x	
							cout << setw(4) << at(ps, pc, py, px) << " ";
							//if (at(ps, pc, py, px) >= 0) { cout << " ";  }
						}
						cout << endl;
					}
					cout << endl;
				}
			}
		}
	}
}; // N C H W // N C H W


class layer
{
public:
	string n_name;
	bool    n_use_gpu;

	float   n_d;    // gradient step size
	layer *n_in1, *n_out1;
	float   leak;
	size2d  n_stride;
	size2d  n_pad;
	float4d n_rsp, n_dif; // layer response, layer chain multiplier
	size2d  n_pool;
	
	layer(void);
	
	void setinput( layer *in_layer ) {
		n_in1 = in_layer;
		in_layer->n_out1 = this;
	}
	void setoutput( layer *out_layer) {
		n_out1 = out_layer;
		out_layer->n_in1 = this;
	}
	void set_toy_rsp( int size_c, int size_h, int size_w, float value);
	void set_rsp( vector<float> labels, int c, int h, int w);
	//void set_rsp(vector<cv::Mat> images);
	//void set_rsp( vector<cv::Mat> images, float min, float max );
	//void show_rsp(string window_name, int db, int channel, float min, float max);
	double avgrsp() {
		double sum = 0;
		for (int t = 0; t < n_rsp.nchw(); t++) { sum += n_rsp(t); }
		return sum / double(n_rsp.nchw());
	}
	double avgrsp(float4d &mask, float mask_value) {
		double sum = 0;
		int tsize = n_rsp.nchw();
		int tcount = 0;
		for (int t = 0; t < tsize; t++) { 
			if ( mask(t) != mask_value) {
				float v = n_rsp(t);
				//if ( fabs(v) > 0.25f ) {
				//	if (v >  1) { v =  1; }
				//	if (v < -1) { v = -1; }
					sum += v ;
					tcount++;
				//}
			}
		}
		if (tcount <= 0) { return 0; }
		return sum / double(tcount);
	}
	
	virtual ~layer(void);
	virtual string layer_type() { return "layer"; }
	virtual void load_init(ifstream &myfile, string layer_type = "" ) {
		if ( layer_type == "" ) {
			myfile >> layer_type;
		}
		string stemp;
		myfile >> stemp; // "size_NCHW"
		int N, C, H, W;
		myfile >> N;
		myfile >> C;
		myfile >> H;
		myfile >> W;
		n_rsp.resize(N, C, H, W);
	}
	virtual void save_init(ofstream &myfile) {
		myfile<<"layer size_NCHW "<<n_rsp.n()<<" "<<n_rsp.c()<<" "<<n_rsp.h()<<" "<<n_rsp.w() << endl ;
	}
	virtual void print( bool print_values = false ) {
		cout << "printing layer " << n_name << endl;
		cout << "n_rsp "; 
		n_rsp.print(print_values);
	}
	virtual void load_weights(ifstream &myfile) {}
	virtual void save_weights(ofstream &myfile) {}
	virtual void set_filter_NCHW(int n, int c, int h, int w, std::mt19937 rng = std::mt19937(0) ) {}
	virtual void set_filter_NCHW(float min = 0.001f, float max = 0.001f, std::mt19937 rng = std::mt19937(0)) {}
	virtual void forward_pass(layer *rsps) {}
	virtual void backward_pass(layer *input, layer *output) {}
	virtual void backward_pass(layer *rsps) {}
	virtual double forward_pass() { return 0;  }
	virtual double backward_pass() { return 0; }
};
