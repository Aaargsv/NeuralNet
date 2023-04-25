#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "layers/layer.h"
#include "layers/activation_layer.h"

class LinearLayer: public ActivationLayer {
public:
    LinearLayer(): ActivationLayer(LayerParameters(LayerType::LINEAR,false)) {}
    float activation(float x) const override;
    void print_info() const override;
    LinearLayer* clone() const override;
};



#endif //LINEAR_LAYER_H
