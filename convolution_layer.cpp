#include "convolution_layer.h"
#include <iostream>

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer &conv_layer): Layer(conv_layer) {
    kernel_size_ = conv_layer.kernel_size_;
    filters_ = conv_layer.filters_;
    padding_ = conv_layer.padding_;
    stride_ = conv_layer.stride_;
}


void ConvolutionLayer::forward(const std::vector<float> &input)  {
    std::cout << "convolution layer forward\n";
    std::cout << "kernel = " << kernel_size_ << std::endl;
    std::cout << "filters = " << filters_ << std::endl;
    std::cout << "padding = " << padding_ << std::endl;
    std::cout << "stride = " << stride_ << std::endl;
    std::cout << "input height = " << h_ << std::endl;
    std::cout << "input weight = " << w_ << std::endl;
    std::cout << "input channels = " << c_ << std::endl;
    std::cout << "vector weights capacity = " << weights_.capacity() << std::endl;
    std::cout << "vector weights real size = " << weights_.size() << std::endl;
    std::cout << "vector weights expected size = " << weights_size_ << std::endl;
    std::cout << "output height = " << out_h_ << std::endl;
    std::cout << "output weight = " << out_w_ << std::endl;
    std::cout << "output channels = " << out_c_ << std::endl;
    std::cout << "vector output capacity = " << output_.capacity() << std::endl;
    std::cout << "vector output real size = " << output_.capacity() << std::endl;
    std::cout << "vector output expected size = " << output_.size() << std::endl;

    std::cout << "----------------------------------------------\n";

}

void ConvolutionLayer::load_weights(std::ifstream &weights_file_input) {
    std::cout << "convolution load weights\n";

}

int ConvolutionLayer::compute_out_height() {
    return (h_ + 2 * padding_ - kernel_size_) / stride_ + 1;
}

int ConvolutionLayer::compute_out_width() {
    return (w_ + 2 * padding_ - kernel_size_) / stride_ + 1;
}

void ConvolutionLayer::setup(int h, int w, int c) {
    h_ = h;
    w_ = w;
    c_ = c;
    input_size_ = h_ * w_ * c_;
    out_h_ = compute_out_height();
    out_w_ = compute_out_width();
    out_c_ = filters_;
    weights_size_ = kernel_size_ * kernel_size_ * filters_;
    output_size_ = out_h_ * out_w_ * out_c_;
    weights_.reserve(weights_size_);
    output_.reserve(output_size_);
}

ConvolutionLayer::~ConvolutionLayer()  {}
