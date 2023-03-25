#include "layers/activation_layer.h"
#include "network.h"

void ActivationLayer::forward(Network &net) {
    std::vector<float> &input = *net.current_tensor;
    for (int i = 0; i < input.size(); i++) {
        input[i] = activation(input[i]);
    }
    outputs_ptr_ = net.current_tensor;
}

int ActivationLayer::setup(const Shape &shape, const Network &net)
{
    in_shape_ = shape;
    out_shape_ = shape;
    return 0;
}

int ActivationLayer::load_pretrained(std::ifstream &input_file)
{
    return 0;
}

const std::vector<float> &ActivationLayer::get_outputs()
{
    return *outputs_ptr_;
}

