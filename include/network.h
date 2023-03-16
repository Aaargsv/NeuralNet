#ifndef NETWORK_H
#define NETWORK_H

#include "neural.h"
#include "layers/layer.h"
#include <vector>
#include <string>

class Network {
public:
    Network(int width, int height, int channels): image_shape(width, height, channels) {}
    ~Network();
    inline void add_layer(Layer *layer) {
        layers_.push_back(layer);
    }
    std::vector<float>* forward(std::vector<float> *input_image);
    void setup();
    int load_pretrained(const std::string &filename);
    friend Network& operator<<(Network &net, const Layer &layer);
protected:
    /// Input image shape
    Shape image_shape;
    /// Network layers
    std::vector<Layer*> layers_;
    std::vector<float> utility_memory;
    std::vector<float> *current_tensor;

    friend class Layer;
    friend class ActivationLayer;
    friend class BatchNormLayer;
    friend class BiasLayer;
    friend class ConvolutionLayer;
    friend class LeakyReluLayer;
    friend class MaxPollingLayer;
    friend class UpsampleLayer;
};
#endif //NETWORK_H
