#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <vector>

class Network {
private:
    //
public:
    //parameters of input tensor
    int w_;
    int h_;
    int c_;

    std::vector<Layer*> layers_;
    std::vector<float> *input_;
    std::vector<float> *output_;

    Network(int width, int height, int channels): w_(width), h_(height), c_(channels) {}

    friend Network& operator<<(Network &net, const Layer &layer);
    void add_layer(Layer *layer);
    void forward(std::vector<float> *input_image);
    void setup();
    ~Network();
};




#endif //NETWORK_H
