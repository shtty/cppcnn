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

class layer_stack
{
private:
	vector<layer*> n_layers;
public:
	layer_stack();
	~layer_stack();

	inline layer & operator[](int i) { return *n_layers[i]; }
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
		while (n_layers.size() > 0) {
			pop_back();
		}
	}
};

