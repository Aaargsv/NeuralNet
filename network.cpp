#include "network.h"

void Network::forward(std::vector<float> *input_image) {
    input_ = input_image;
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i].forward(*input_);
        input_ = &layers_[i].output_;
    }
}

void Network::add_layer(Layer *layer) {
    layers_.push_back(layer);
}
