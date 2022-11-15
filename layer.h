#ifndef LAYER_H
#define LAYER_H

#include "neural.h"
#include "network.h"
#include <vector>
#include <fstream>


class Layer {
private:
    //
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

    /*layer characteristic*/
    LayerType layer_type_;

    /*output feature maps*/
    std::vector<float> output_;

    Layer(LayerType layer_type):layer_type_(layer_type) {}
    Layer(Layer &layer) {
        layer_type_ = layer.layer_type_;
    }

    virtual void forward(const std::vector<float> &input)  = 0;
    virtual Layer* clone() const = 0;
    virtual void load_weights(std::ifstream &weights_file_input) = 0;
    virtual ~Layer();
protected:
    std::vector<float> weights_;
};

Network& operator<<(Network &net, const Layer &layer) {
    Layer *layer_ptr = layer.clone();
    net.add_layer(layer_ptr);
    return net;
}


#endif //LAYER_H
