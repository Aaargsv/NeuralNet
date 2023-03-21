#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H

#include "layers/layer.h"
#include "layers/activation_layer.h"

class LogisticLayer: public ActivationLayer {
public:
    LogisticLayer(): LogisticLayer(LayerParameters(LayerType::LOGISTIC,false)) {}
    float activation(float x) const override;
    void print_info() const override;
    LogisticLayer* clone() const override;
};

#endif //LOGISTIC_LAYER_H
