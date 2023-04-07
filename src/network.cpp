#include "network.h"
#include "layers/yolo_layer.h"
#include <fstream>
#include <algorithm>


Network::~Network()
{
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
    }
}

std::vector<float>* Network::forward(std::vector<float>* input_image)
{
    current_tensor = input_image;
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->forward(*this);
    }
    return current_tensor;
}

int Network::setup()
{
    Shape input_layer_shape = net_shape_;
    int utility_memory_size = 0;
    int temp;
    for (int i = 0; i < layers_.size(); i++) {
        if ((temp = layers_[i]->setup(input_layer_shape, *this)) < 0) {
            return 1;
        }
        utility_memory_size = std::max(temp, utility_memory_size);
        input_layer_shape = layers_[i]->out_shape();
    }
    utility_memory.reserve(utility_memory_size);
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

void Network::gather_bounding_boxes()
{
    bounding_boxes_.reserve(num_classes_);
    for (int i = 0; i < layers_.size(); i++) {
        if (layers_[i]->layer_param().layer_type_ == LayerType::YOLO) {
            static_cast<YoloLayer*>(layers_[i])->get_bounding_boxes(bounding_boxes_,
                                                                    net_shape_.h, net_shape_.w);
        }
    }
}

void Network::nms()
{

}

Network& operator<<(Network &net, const Layer &layer)
{
    net.add_layer(layer.clone());
    return net;
}


