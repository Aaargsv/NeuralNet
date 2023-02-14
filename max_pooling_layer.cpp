#include "max_pooling_layer.h"
#include "utils.h"
#include <iostream>

void MaxPollingLayer::forward(std::vector<float> *input_tensor,
                              std::vector<float> *output_tensor) {
    print_info();
}

void MaxPollingLayer::setup(const Shape &shape) {
    in_shape_ = shape;
    out_shape_.reshape(compute_out_height(), compute_out_width(), shape.c_);
    outputs_.reserve(out_shape_.get_size());
}

void MaxPollingLayer::print_info() const {
    std::cout << "LAYER NAME: MAX_POOLING\n";
    std::cout << "WINDOW SIZE: "    << window_size_ << "\n";
    std::cout << "PADDING: "        << padding_ << "\n";
    std::cout << "stride: "         << stride_ << "\n";
    std::cout << "INPUT TENSOR: "   << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: "  << out_shape_ << "\n";
    std::cout << "------------------------\n";
}



