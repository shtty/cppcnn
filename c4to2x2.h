#pragma once
#include "layer.h"
class c4to2x2 : public layer
{
public:
	string layer_type() { return "c4to2x2"; }
	c4to2x2(void);
	~c4to2x2(void);
	void load_init(ifstream &myfile, string layer_type = "" );
	void save_init(ofstream &myfile);
	
	double forward_pass();
	double backward_pass();
};

