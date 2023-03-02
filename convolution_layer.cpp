#include "convolution_layer.h"
#include "utils.h"
#include <iostream>

void ConvolutionLayer::forward(std::vector<float> *input_tensor, std::vector<float> *output_tensor) {
    print_info();
    inter_layer->print_info();
}

void ConvolutionLayer::setup(const Shape &shape) {
    in_shape_ = shape;
    weights_length_ = kernel_size_ * kernel_size_ * filters_;
    out_shape_.reshape(compute_out_height(), compute_out_width(), filters_);
    weights_.reserve(weights_length_);
    outputs_.reserve(out_shape_.get_size());
    if (has_batch_norm_)
        inter_layer = new BatchNormLayer();
    else
        inter_layer = new BiasLayer();
    inter_layer->setup(out_shape_);
}

int ConvolutionLayer::load_pretrained(std::ifstream &weights_file) {

    inter_layer->load_pretrained(weights_file);
    weights_file.read(reinterpret_cast<char*>(weights_.data()),
                    kernel_size_ * kernel_size_ * filters_);
    if (weights_file.gcount() / sizeof(float) != weights_length_)
        return 1;
    std::cout << "convolution load weights\n";
    return 0;
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





