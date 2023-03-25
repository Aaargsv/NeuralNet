#include "layers/leaky_relu_layer.h"
#include "operations/activation_functions.h"
#include <iostream>


float LeakyReluLayer::activation(float x) const
{
    return leaky_relu(x);
}

void LeakyReluLayer::print_info() const
{
    std::cout << "LAYER NAME: LEAKY ReLU\n";
    std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
    std::cout << "------------------------\n";
}

LeakyReluLayer *LeakyReluLayer::clone() const
{
    return new LeakyReluLayer(*this);
}