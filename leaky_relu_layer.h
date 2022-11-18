#ifndef LEAKY_RELU_LAYER_H
#define LEAKY_RELU_LAYER_H

#include "layer.h"
#include "activation_layer.h"

class LeakyReluLayer: public ActivationLayer {
private:
    //
public:
    LeakyReluLayer(): ActivationLayer(LayerType::LEAKY_ReLU) {}
    LeakyReluLayer(const LeakyReluLayer &leaky_ReLU_layer): ActivationLayer(leaky_ReLU_layer) {}

    float activation(float x) const override {
        return x > 0 ? x : 0.1 * x;
    }

    LeakyReluLayer* clone() const override {
        return new LeakyReluLayer(*this);
    }

    ~LeakyReluLayer() override { }
};
#endif //LEAKY_RELU_LAYER_H
