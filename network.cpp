#include "network.h"

void Network::forward(std::vector<float> *input_image) {
    input_ = input_image;
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->forward(*input_);
        input_ = &layers_[i]->output_;
    }
}

void Network::add_layer(Layer *layer) {
    layers_.push_back(layer);
}

void Network::setup() {
    int w = w_;
    int h = h_;
    int c = c_;
    int N = layers_.size();
    for (int i = 0; i < N; i++) {
        layers_[i]->setup(w, h, c);
        w = layers_[i]->out_w_;
        h = layers_[i]->out_h_;
        c = layers_[i]->out_c_;
    }
    output_ = &layers_[N - 1]->output_;
}

Network::~Network() {
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
    }
}

Network& operator<<(Network &net, const Layer &layer) {
    Layer *layer_ptr = layer.clone();
    net.add_layer(layer_ptr);
    return net;
}


