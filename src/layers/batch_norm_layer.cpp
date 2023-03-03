#include "layers/batch_norm_layer.h"
#include "operations/tensor_math.h"
#include <iostream>

void BatchNormLayer::forward(std::vector<float> *input_tensor,
                             std::vector<float> *output_tensor)
{
    output_tensor = input_tensor;
    print_info();
}

int BatchNormLayer::load_pretrained(std::ifstream &weights_file)
{
    if (!weights_file.read(reinterpret_cast<char*>(beta_.data()),
                           out_shape_.c_ * sizeof(float)))
        return 1;
    if (!weights_file.read(reinterpret_cast<char*>(gamma_.data()),
                           out_shape_.c_ * sizeof(float)))
        return 1;
    if (!weights_file.read(reinterpret_cast<char*>( rolling_mean_.data()),
                           out_shape_.c_ * sizeof(float)))
        return 1;
    if (!weights_file.read(reinterpret_cast<char*>( rolling_variance_.data()),
                           out_shape_.c_ * sizeof(float)))
        return 1;
    return 0;
}

void BatchNormLayer::setup(const Shape &shape)
{
    in_shape_ = shape;
    out_shape_ = shape;
    rolling_mean_.reserve(shape.c_);
    rolling_variance_.reserve(shape.c_);
    gamma_.reserve(shape.c_);
    beta_.reserve(shape.c_);
}

void BatchNormLayer::print_info() const
{
    std::cout << "LAYER NAME: BATCH_NORM\n";
    std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
    std::cout << "------------------------\n";
}
