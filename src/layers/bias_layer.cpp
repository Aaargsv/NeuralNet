#include "layers/bias_layer.h"
#include "network.h"
#include <iostream>

void BiasLayer::forward(Network &net)
{
    std::vector<float> &input = *net.current_tensor;
    add_bias(input, bias_, in_shape_.c_, in_shape_.h_ * in_shape_.w_);
    ouputs_ptr = net.current_tensor;
}

int BiasLayer::setup(const Shape &shape, const Network &net)
{
    in_shape_ = shape;
    out_shape_ = shape;
    bias_.reserve(shape.c_);
    return 0;
}

int BiasLayer::load_pretrained(std::ifstream &weights_file)
{
    if(!weights_file.read(reinterpret_cast<char*>(bias_.data()), out_shape_.c_ * sizeof(float)))
        return 1;
    return 0;
}

const std::vector<float> &BiasLayer::get_outputs()
{
    return *ouputs_ptr;
}

void BiasLayer::print_info() const
{
    std::cout << "LAYER NAME: BIAS\n";
    std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
    std::cout << "------------------------\n";
}

inline BiasLayer *BiasLayer::clone() const
{
    return new BiasLayer(*this);
};