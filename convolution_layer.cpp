#include "convolution_layer.h"
#include "utils.h"
#include <iostream>

void ConvolutionLayer::forward(std::vector<float> *input_tensor, std::vector<float> *output_tensor) {
    print_info();
    inter_layer->print_info();
}

void ConvolutionLayer::setup(const Shape &shape) {
    in_shape_ = shape;
    out_shape_.reshape(compute_out_height(), compute_out_width(), filters_);
    weights_.reserve(kernel_size_ * kernel_size_ * filters_);
    outputs_.reserve(out_shape_.get_size());

    if (has_batch_norm_)
        inter_layer = new BatchNormLayer();
    else
        inter_layer = new BiasLayer();
    inter_layer->setup(out_shape_);
}

void ConvolutionLayer::load_pretrained(std::ifstream &input_file) {
    inter_layer->load_pretrained(input_file);
    std::cout << "convolution load weights\n";
}

void ConvolutionLayer::print_info() const {
    std::cout << "LAYER NAME: CONVOLUTION\n";
    std::cout << "KERNEL SIZE: "    << kernel_size_ << "\n";
    std::cout << "FILTERS: "        << filters_ << "\n";
    std::cout << "PADDING: "        << padding_ << "\n";
    std::cout << "STRIDE: "         << stride_ << "\n";
    std::cout << "INPUT TENSOR: "   << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: "  << out_shape_ << "\n";
    std::cout << "------------------------\n";
}





