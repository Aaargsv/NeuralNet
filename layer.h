#ifndef LAYER_H
#define LAYER_H

#include "neural.h"
#include <vector>
#include <fstream>

class Layer {
private:

public:
    /*feature maps sizes*/
    int h_;
    int w_;
    int c_;
    int out_h_;
    int out_w_;
    int out_c_;
    int input_size_;
    int output_size_;
    int weights_size_;
    bool can_weights_be_loaded_;

    /*layer characteristic*/
    LayerType layer_type_;

    /*output feature maps*/
    std::vector<float> output_;

    Layer(LayerType layer_type, bool can_weights_be_loaded):layer_type_(layer_type),
                                                            can_weights_be_loaded_(can_weights_be_loaded) {}
    Layer(const Layer &layer) {
        layer_type_ = layer.layer_type_;
        can_weights_be_loaded_ = layer.can_weights_be_loaded_;
    }

    virtual void forward(const std::vector<float> &input)  = 0;
    virtual Layer* clone() const = 0;
    virtual void setup(int h, int w, int c) = 0;
    virtual void load_weights(std::ifstream &weights_file_input) = 0;
    virtual ~Layer() {};
protected:
    std::vector<float> weights_;
};

#endif //LAYER_H
