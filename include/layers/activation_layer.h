#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"
#include "neural.h"

class ActivationLayer: public Layer {
public:
    ActivationLayer(const LayerParameters &layer_param): Layer(layer_param) {}
    ~ActivationLayer() {}
    void forward(Network &net) override;
    int setup(const Shape &shape) override;

protected:
    virtual float activation(float x) const = 0;
private:
    int load_pretrained(std::ifstream &input_file) override;
};

#endif //ACTIVATION_LAYER_H
