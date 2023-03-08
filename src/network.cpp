#include "layers/network.h"
#include <fstream>
Network::~Network()
{
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
    }
}

std::vector<float>* Network::forward(std::vector<float>* input_image)
{
    std::vector<float> *tensor;
    tensor = layers_[0]->forward(input_image);
    for (int i = 1; i < layers_.size(); i++) {
        tensor = layers_[i]->forward(tensor);
    }
    return tensor;
}

void Network::setup()
{
    Shape input_layer_shape = image_shape;
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->setup(input_layer_shape);
        input_layer_shape = layers_[i]->out_shape();
    }
}

int Network::load_pretrained(const std::string &filename)
{
    std::ifstream file_weights(filename);
    if (!file_weights) {
        return 1;
    }
    for (int i = 0; i < layers_.size(); i++) {
        if (layers_[i]->load_pretrained(file_weights)) {
            return 1;
        }
    }
    return 0;
}

Network& operator<<(Network &net, const Layer &layer)
{
    net.add_layer(layer.clone());
    return net;
}


