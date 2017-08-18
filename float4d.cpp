#include "float4d.h"

std::mt19937 float4d::n_random_seed = std::mt19937(0);

void float4d::set(float min, float max, float cmin, float cmax, float multi, float add) {
	float4d r;
	r.resize(n_size);
	if (cmin >= cmax) {
		cmax = *max_element(p_v.begin(), p_v.end());
		cmin = *min_element(p_v.begin(), p_v.end());
	}
	for (int n = 0; n < n_size.n; n++) {
		for (int c = 0; c < n_size.c; c++) {
			for (int h = 0; h < n_size.h; h++) {
				for (int w = 0; w < n_size.w; w++) {
					float v = this->operator()(n, c, h, w);
					r(n, c, h, w) = ((v / (cmax - cmin))*(max - min) + min)*multi + add;
				}
			}
		}
	}
	*this = r;
}
void float4d::set(string init_method, std::mt19937 &rng) {
	if (init_method == "xavier" || init_method == "Xavier") {
		////http://deepdish.io/2015/02/24/network-initialization/
		//// X.Glorot and Y.Bengio, “Understanding the difficulty of training deep feedforward neural networks, ” in International conference on artificial intelligence and statistics, 2010, pp. 249–256.
		//// Xavier initialization
		float d = sqrt(12 / double(chw())) / 2;
		float max = d;
		float min = -d;
		set(min, max,rng);
	}
	if (init_method == "pass") {
		set("xavier");
		int ch = n_size.h / 2;
		int cw = n_size.w / 2;
		set(0, 0);
		this->operator()(0, 0, ch, cw) = 1;
	}
	
}
void float4d::set_borders(int border_width, float border_value) {
	for (int n = 0; n < n_size.n; n++) {
		for (int c = 0; c < n_size.c; c++) {
			for (int h = 0; h < n_size.h; h++) {
				for (int w = 0; w < n_size.w; w++) {
					if (h < border_width || h >= n_size.h - border_width ||
						w < border_width || w >= n_size.w - border_width) {
						this->operator()(n, c, h, w) = border_value;
					}
				}
			}
		}
	}
}
void float4d::print(bool print_values) {
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

void float4d::rotate(int cy, int cx, int angle_degree, float out_bound_value) {
	float cosangle = std::cos(float(-angle_degree * PI / 180.0));
	float sinangle = std::sin(float(-angle_degree * PI / 180.0));

	float4d r;
	r.resize(this->size());
	r.set(out_bound_value);
	for (int ps = 0; ps < n_size.n; ps++) { // per each image
		for (int pc = 0; pc < n_size.c; pc++) { // per each channel
			for (int py = 0; py < n_size.h; py++) { // per each y 
				for (int px = 0; px < n_size.w; px++) { // per each x	
					float y, x, ry, rx;
					y = cy - py;
					x = px - cx;
					rx = cosangle*x - sinangle*y;
					ry = sinangle*x + cosangle*y;
					int rpy, rpx;
					rpy = int(cy - ry + 0.5f);
					rpx = int(rx + cx + 0.5f);
					if (rpy >= 0 && rpy < n_size.h && rpx >= 0 && rpx < n_size.w) {
						r(ps, pc, py, px) = this->p_v[nchw2idx(ps, pc, rpy, rpx)];
					}

				}
			}
		}
	}
	this->p_v = r.p_v;
}
void float4d::translate(int th, int tw, float out_bound_value ) {
	float4d r;
	r.resize(this->size());
	r.set(out_bound_value);
	for (int ps = 0; ps < n_size.n; ps++) { // per each image
		for (int pc = 0; pc < n_size.c; pc++) { // per each channel
			for (int py = 0; py < n_size.h; py++) { // per each y 
				for (int px = 0; px < n_size.w; px++) { // per each x
					int ny, nx;
					ny = py + th;
					nx = px + tw;
					if (ny >= 0 && nx >= 0 && ny < n_size.h && nx < n_size.w) {
						r(ps, pc, py, px) = this->p_v[nchw2idx(ps, pc, ny, nx)];
					}
				}
			}
		}
	}
	this->p_v = r.p_v;
}
