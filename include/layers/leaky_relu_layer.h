#ifndef LEAKY_RELU_LAYER_H
#define LEAKY_RELU_LAYER_H

#include "layer.h"
#include "activation_layer.h"

class LeakyReluLayer: public ActivationLayer {
public:
    LeakyReluLayer(): ActivationLayer(LayerParameters(LayerType::LEAKY_ReLU,false)) {}

    inline float activation(float x) const override {
        return x > 0 ? x : 0.1 * x;
    }
    inline void print_info() const override {
        std::cout << "LAYER NAME: LEAKY ReLU\n";
        std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
        std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
        std::cout << "------------------------\n";
    }
    inline LeakyReluLayer* clone() const override {
        return new LeakyReluLayer(*this);
    }

};
#endif //LEAKY_RELU_LAYER_H
