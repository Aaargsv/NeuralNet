#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"
#include "neural.h"
#include <iostream>

class ActivationLayer: public Layer {
public:
    ActivationLayer(const LayerParameters &layer_param): Layer(layer_param) {}
    ~ActivationLayer() {}
    inline void forward(std::vector<float> *input_tensor, std::vector<float> *output_tensor) override {
        std::vector<float> &input = *input_tensor;
        std::vector<float> &output = *output_tensor;
        for (int i = 0; i < input.size(); i++) {
            output[i] = activation(input[i]);
        }
    }

    inline void setup(const Shape &shape) override {
        in_shape_ = shape;
        out_shape_ = shape;
    }

protected:
    virtual float activation(float x) const = 0;
private:
    int load_pretrained(std::ifstream &input_file) override
    {
        return 0;
    }
};



#endif //ACTIVATION_LAYER_H
