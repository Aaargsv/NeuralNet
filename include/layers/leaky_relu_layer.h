#ifndef LEAKY_RELU_LAYER_H
#define LEAKY_RELU_LAYER_H

#include "layer.h"
#include "activation_layer.h"

class LeakyReluLayer: public ActivationLayer {
public:
    LeakyReluLayer(): ActivationLayer(LayerParameters(LayerType::LEAKY_ReLU,false)) {}
    float activation(float x) const override;
    void print_info() const override;
    LeakyReluLayer* clone() const override;
};
#endif //LEAKY_RELU_LAYER_H
