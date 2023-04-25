#include "layers/linear_layer.h"

#include <iostream>


float LinearLayer::activation(float x) const
{
    return x;
}

void LinearLayer::print_info() const
{
    std::cout << "LAYER NAME: LINEAR ReLU\n";
    std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
    std::cout << "------------------------\n";
}

LinearLayer *LinearLayer::clone() const
{
    return new LinearLayer(*this);
}
