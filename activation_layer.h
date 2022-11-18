#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"
#include <iostream>

class ActivationLayer: public Layer {
private:
    void load_weights(std::ifstream& weights_file_input) override {}
public:
    virtual float activation(float x) const = 0;

    ActivationLayer(LayerType layer_type): Layer(layer_type, false) {}
    ActivationLayer(const ActivationLayer &activation_layer): Layer(activation_layer) {}

    void forward(const std::vector<float>& input) override {
        std::cout << "activation forward\n";
        std::cout << "input height = " << h_ << std::endl;
        std::cout << "input weight = " << w_ << std::endl;
        std::cout << "input channels = " << c_ << std::endl;
        std::cout << "output height = " << out_h_ << std::endl;
        std::cout << "output weight = " << out_w_ << std::endl;
        std::cout << "output channels = " << out_c_ << std::endl;
        std::cout << "----------------------------------------------\n";

        /*output_ = &input;
        for (int i = 0; i < input.size(); i++) {
            (*output_)[i] = activation(input[i]);
        }*/
    }

    void setup(int h, int w, int c) override {
        h_ = h;
        w_ = w;
        c_ = c;
        input_size_ = h_ * w_ * c_;
        out_h_ = h;
        out_w_ = w;
        out_c_ = c;
        output_size_ = h * w * c;
    }
};

#endif //ACTIVATION_LAYER_H
