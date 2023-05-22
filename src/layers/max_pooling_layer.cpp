#include "layers/max_pooling_layer.h"
#include "operations/max_pool.h"
#include "utils/utils.h"
#include "network.h"
#include <iostream>

void MaxPollingLayer::forward(Network &net) {
    std::vector<float> &input = *net.current_tensor;

    /*max_pool(input, in_shape_.c, in_shape_.h, in_shape_.w,
             window_size_, stride_, padding_, outputs_);*/

    max_pool2(input, in_shape_.c, in_shape_.h, in_shape_.w,
              window_size_, stride_, padding_, outputs_);

    net.current_tensor = &outputs_;
    //print_info();
}

int MaxPollingLayer::setup(const Shape &shape, const Network &net) {
    in_shape_ = shape;
    out_shape_.reshape(compute_out_height(), compute_out_width(), shape.c);
    std::cout << "[MaxPolling] " << "(" << window_size_ << ", " << padding_ << ", " << stride_ << ") :"
                << out_shape_.h << std::endl;

    outputs_.resize(out_shape_.get_size());
    return 0;
}

const std::vector<float> &MaxPollingLayer::get_outputs()
{
    return outputs_;
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

MaxPollingLayer *MaxPollingLayer::clone() const
{
    return new MaxPollingLayer(*this);
}
int MaxPollingLayer::compute_out_width() const
{
    return (in_shape_.w + padding_ - window_size_) / stride_ + 1;
}
int MaxPollingLayer::compute_out_height() const
{
    return (in_shape_.h + padding_ - window_size_) / stride_ + 1;
}

int MaxPollingLayer::load_pretrained(std::ifstream &weights_file, std::ofstream &check_file)
{
    return 0;
};


