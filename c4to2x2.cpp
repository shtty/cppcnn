////////////////////////////////////////////////////////////////////////////////////
////		This code is written by Ho Yub Jung                                 ////
////////////////////////////////////////////////////////////////////////////////////

#include "c4to2x2.h"


c4to2x2::c4to2x2(void)
{
	n_name = "c4to2x2";
}


c4to2x2::~c4to2x2(void){
}

void c4to2x2::load_init(ifstream &myfile, string layer_type ) {
	if (layer_type == "") {
		myfile >> layer_type;
	}
}
void c4to2x2::save_init(ofstream &myfile) {
	myfile << endl;
	myfile << "c4to2x2" << endl;
}

double c4to2x2::forward_pass() {
	int nsize = p_in1->n_rsp.n();
	int csize = p_in1->n_rsp.c() / 4;
	if (p_in1->n_rsp.c() % 4 > 0) { csize++;  }
	int hsize = p_in1->n_rsp.h() * 2;
	int wsize = p_in1->n_rsp.w() * 2;

	n_rsp.resize(nsize, csize, hsize, wsize);
	n_rsp.set(0);

	for (int pn = 0; pn < p_in1->n_rsp.n(); pn++) { // per each image sample
	for (int pc = 0; pc < p_in1->n_rsp.c(); pc++) { // per each channel
	for (int ph = 0; ph < p_in1->n_rsp.h(); ph++) { // per each y
	for (int pw = 0; pw < p_in1->n_rsp.w(); pw++) { // per each x
		int c, h, w;
		c = pc / 4;
		if (pc % 4 == 0) { h = 2 * ph + 0;  w = 2 * pw + 0; }
		if (pc % 4 == 1) { h = 2 * ph + 0;  w = 2 * pw + 1; }
		if (pc % 4 == 2) { h = 2 * ph + 1;  w = 2 * pw + 0; }
		if (pc % 4 == 3) { h = 2 * ph + 1;  w = 2 * pw + 1; }
		n_rsp(pn, c, h, w) = p_in1->n_rsp(pn, pc, ph, pw);
	}}}}

	return 0;
}


double c4to2x2::backward_pass() {
	n_dif.resize(p_in1->n_rsp.size());
	for (int pn = 0; pn < n_dif.n(); pn++) { // per each image sample
	for (int pc = 0; pc < n_dif.c(); pc++) { // per each channel
	for (int ph = 0; ph < n_dif.h(); ph++) { // per each y
	for (int pw = 0; pw < n_dif.w(); pw++) { // per each x
		int c, h, w;
		c = pc / 4;
		if (pc % 4 == 0) { h = 2 * ph + 0;  w = 2 * pw + 0; }
		if (pc % 4 == 1) { h = 2 * ph + 0;  w = 2 * pw + 1; }
		if (pc % 4 == 2) { h = 2 * ph + 1;  w = 2 * pw + 0; }
		if (pc % 4 == 3) { h = 2 * ph + 1;  w = 2 * pw + 1; }
		n_dif(pn, pc, ph, pw) = p_out1->n_dif(pn, c, h, w);
	}}}}
	return 0;
}