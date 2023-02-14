#include "batch_norm_layer.h"
#include <iostream>

void BatchNormLayer::forward(std::vector<float> *input_tensor,
                             std::vector<float> *output_tensor) {
    print_info();
}

void BatchNormLayer::load_pretrained(std::ifstream &input_file) {

}

void BatchNormLayer::setup(const Shape &shape) {
    in_shape_ = shape;
    out_shape_ = shape;
    rolling_mean_.reserve(shape.c_);
    rolling_variance_.reserve(shape.c_);
    gamma_.reserve(shape.c_);
    beta_.reserve(shape.c_);
}

void BatchNormLayer::print_info() const  {
    std::cout << "LAYER NAME: BATCH_NORM\n";
    std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
    std::cout << "------------------------\n";
}
