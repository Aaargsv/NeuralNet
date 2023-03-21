#include "layers/logistic_layer.h"
#include <cmath>

float LogisticLayer::activation(float x) const
{
    return 1 / (1 + std::exp(-x));
}

void LogisticLayer::print_info() const
{
    std::cout << "LAYER NAME: LEAKY ReLU\n";
    std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
    std::cout << "------------------------\n";
}

LogisticLayer *LogisticLayer::clone() const
{
    return new LeakyReluLayer(*this);
}