////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////
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
#include <math.h>
using namespace std;

#define PI 3.14159265

class size4d {
public:
	int n, c, h, w;
	size4d(int in = 0, int ic = 0, int ih = 0, int iw = 0) { set(in, ic, ih, iw); }
	inline void set(int in, int ic, int ih, int iw) {
		n = in; c = ic;  h = ih; w = iw;
	}
	inline int nchw() { return  n*c*h*w; }
	inline bool operator!=(size4d in) {
		return !(n == in.n && c == in.c && h == in.h && w == in.w);
	}
	bool operator==( size4d other) {
		return n == other.n && c == other.c && h == other.h && w == other.w;
	}
};
class size2d {
public:
	int w, h;
	size2d() { w = h = 0; }
	inline void set(int ih, int iw) { h = ih; w = iw; }
};
class float1d {
private:
	int n_size;
	vector<float> p_v;
public:
	float1d() {}
	~float1d() {}
	inline void resize(int new_size) {
		if (new_size != p_v.size()) {
			p_v.resize(new_size);
		}
		n_size = new_size;
	}
	inline void set(float v) { std::fill(p_v.begin(), p_v.end(), v); }
	inline int size() { return n_size; }
	inline float & at(int n) { return p_v[n]; }
	inline float & operator()(int n) { return p_v[n]; }
	void print(bool print_values = false) {
		cout << "size1d " << n_size << ": ";
		if (print_values) {
			for (int i = 0; i < n_size; i++) { cout << at(i) << " "; }
		}
		cout << endl;
	}
	void set(float min, float max, std::mt19937 &rng = std::mt19937(0)) {
		std::uniform_real_distribution<float> uniform_dist(min, max);
		for (int p = 0; p < n_size; p++) {
			p_v[p] = uniform_dist(rng);
		}
	}
};

class float4d { // N C H W
private:
	size4d n_size;
	vector<float> p_v;
	int xw, xwxh, xwxhxc, xwxhxcxn;
	float n_zero;
public:	
	static std::mt19937 n_random_seed;
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
	inline float & operator()(int n, int c, int p) { return p_v[n*xwxhxc + c*xwxh + p]; }
	float4d & operator+=(float v) {
		for (int i = 0; i < xwxhxcxn; i++) {
			p_v[i] += v;
		}
		return *this;
	}
	bool operator==( float4d &other) {
		if (n_size != other.size()) {
			return false;
		}
		for (int p = 0; p < xwxhxcxn; p++ ) {
			if ( p_v[p] != other[p] ) {
				return false;
			}
		}
		return true;
	}

	inline size4d size() { return n_size; }
	inline int nchw() { return xwxhxcxn; }
	inline int  chw() { return xwxhxc  ; }
	inline int   hw() { return xwxh; }
	inline int w() { return n_size.w;  }
	inline int h() { return n_size.h; }
	inline int c() { return n_size.c; }
	inline int n() { return n_size.n; }
	
	inline void set(float value, int nidx) { std::fill(p_v.begin() + nidx*xwxhxc, p_v.begin() + nidx*xwxhxc + xwxhxc, value); }
	inline void set(float value, int nidx, int cidx) { std::fill(p_v.begin() + nidx*xwxhxc + cidx*xwxh, p_v.begin() + nidx*xwxhxc + cidx*xwxh + xwxh, value); }
	inline void set(float value) { std::fill(p_v.begin(), p_v.end(), value);  }
	inline float max() { return *max_element(p_v.begin(), p_v.end()); }
	inline float min() { return *min_element(p_v.begin(), p_v.end()); }
	void set(float min, float max, std::mt19937 &rng = n_random_seed) {
		std::uniform_real_distribution<float> uniform_dist(min, max);
		for (int p = 0; p < xwxhxcxn; p++) {
			p_v[p] = uniform_dist(rng);
		}
	}
	float sum() {
		double allsum = 0;
		for (int p = 0; p < xwxhxcxn; p++) { allsum += p_v[p]; }
		return float(allsum);
	}
	float avg(float4d weights = float4d()) {
		if (weights.n_size == this->n_size) {
			double allsum = 0;
			double allweight = 0;
			for (int p = 0; p < xwxhxcxn; p++) {
				allsum += p_v[p] * weights[p];
				allweight += weights[p];
			}
			return float(allsum / allweight);
		}
		return sum() / float(this->xwxhxcxn);
	}
	void normalize( string method = "minmax") {
		if ( method == "minmax" ) {
			float cmin, cmax; ///// Minus Min, Divided by Max
			cmin = this->min();
			for (int p = 0; p < xwxhxcxn; p++) { p_v[p] -= cmin; }
			cmax = this->max();
			if ( cmax > 0 ) {
				cmax = 1 / cmax;
				for (int p = 0; p < xwxhxcxn; p++) { p_v[p] *= cmax; }
			}
		}
		else if ( method.find("std") != string::npos ) {
			double mean = avg();
			double sqr_sum = 0;
			for (int p = 0; p < xwxhxcxn; p++) {
				sqr_sum += (p_v[p] -mean)*(p_v[p] - mean);
			}
			double std = sqrt(sqr_sum / double(xwxhxcxn));
			for (int p = 0; p < xwxhxcxn; p++) {
				p_v[p] = float( (p_v[p] - mean) / std );
			}

		}
		else {
			cout << "float4d::normalize: no implementation of normalize( " << method << " )" << endl;
		}
	}
	void truncate( float min = 0, float max = 1) {
		for (int p = 0; p < xwxhxcxn; p++) {
			if (p_v[p] < min) { p_v[p] = min; }
			if (p_v[p] > max) { p_v[p] = max; }
		}

	}
	void set(float min, float max, float cmin, float cmax, float multi, float add);
	void set(string init_method = "xavier", std::mt19937 &rng = n_random_seed);
	void set_borders(int border_width, float border_value);
	
	void print(bool print_values = false);
	void rotate(int cy, int cx, int angle_degree, float out_bound_value = 0);
	void translate(int th = 0, int tw = 0, float out_bound_value = 0);
}; // N C H W // N C H W


