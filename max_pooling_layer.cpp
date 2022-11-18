#include "max_pooling_layer.h"
#include <iostream>

void MaxPollingLayer::forward(const std::vector<float> &input) {
    std::cout << "max_pooling_layer\n";
    std::cout << "window_size = " << window_size_ << std::endl;
    std::cout << "padding = " << padding_ << std::endl;
    std::cout << "stride = " << stride_ << std::endl;
    std::cout << "input height = " << h_ << std::endl;
    std::cout << "input weight = " << w_ << std::endl;
    std::cout << "input channels = " << c_ << std::endl;
    std::cout << "output height = " << out_h_ << std::endl;
    std::cout << "output weight = " << out_w_ << std::endl;
    std::cout << "output channels = " << out_c_ << std::endl;
    std::cout << "vector output capacity = " << output_.capacity() << std::endl;
    std::cout << "vector output real size = " << output_.capacity() << std::endl;
    std::cout << "vector output expected size = " << output_.size() << std::endl;
    std::cout << "----------------------------------------------\n";
}

int MaxPollingLayer::compute_out_height() {
    return (h_ + padding_ - window_size_) / stride_ + 1;
}

int MaxPollingLayer::compute_out_width() {
    return (w_ + padding_ - window_size_) / stride_ + 1;
}

void MaxPollingLayer::setup(int h, int w, int c) {
    h_ = h;
    w_ = w;
    c_ = c;
    input_size_ = h * w * c;
    out_h_ = compute_out_height();
    out_w_ = compute_out_width();
    out_c_ = c;
    output_size_ = out_h_ * out_w_ * out_c_;
    output_.reserve(output_size_);
}

MaxPollingLayer* MaxPollingLayer::clone() const {
    return new MaxPollingLayer(*this);
};

MaxPollingLayer::~MaxPollingLayer() {}

