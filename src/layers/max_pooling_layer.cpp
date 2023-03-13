#include "layers/max_pooling_layer.h"
#include "operations/max_pool.h"
#include "utils/utils.h"
#include <iostream>

std::vector<float> *MaxPollingLayer::forward(std::vector<float> *input_tensor,
                                             std::vector<float> &utility_memory) {
    std::vector<float> &input = *input_tensor;
    max_pool(input, in_shape_.c_, in_shape_.h_, in_shape_.w_,
             window_size_, stride_, padding_, outputs_);
    print_info();
    return &outputs_;
}

int MaxPollingLayer::setup(const Shape &shape) {
    in_shape_ = shape;
    out_shape_.reshape(compute_out_height(), compute_out_width(), shape.c_);
    outputs_.reserve(out_shape_.get_size());
    return 0;
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



