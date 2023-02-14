#include "network.h"

Network::~Network() {
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
    }
}

std::vector<float>* Network::forward(std::vector<float> *input_image) {
    std::vector<float> *input_tensor = input_image;
    std::vector<float> *output_tensor;
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->forward(input_tensor, output_tensor);
        input_tensor = output_tensor;
    }
    return output_tensor;
}

void Network::setup() {
    Shape input_layer_shape = image_shape;
    int N = layers_.size();
    for (int i = 0; i < N; i++) {
        layers_[i]->setup(input_layer_shape);
        input_layer_shape = layers_[i]->out_shape();
    }
}

void Network::load_pretrained(std::string &filename) {

}

Network& operator<<(Network &net, const Layer &layer) {
    net.add_layer(layer.clone());
    return net;
}


