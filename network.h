#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <vector>

class Network {
private:
    //
public:
    //parameters of input image
    int w_;
    int h_;
    int c_;

    std::vector<Layers*> layers_;
    std::vector<float> *input_;
    std::vector<float> *output_;

    Network(int width, int height, int channels): w_(width), h_(height), c_(channels) {}
    void add_layer(Layer *layer);
    void forward(std::vector<float> *input_image);

};


#endif //NETWORK_H
