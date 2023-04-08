#ifndef NETWORK_H
#define NETWORK_H

#include "neural.h"
#include "layers/layer.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <string>

class Network {
public:
    Network(const std::string &image_file_name, const std::string &weight_file_name, int &error_status);
    ~Network();

    inline void add_layer(Layer *layer) {
        layers_.push_back(layer);
    }
    std::vector<float>* forward(std::vector<float> *input_image);
    int setup();
    int load_image(const std::string &filename);
    int load_pretrained(const std::string &filename);
    void gather_bounding_boxes();
    void apply_nms(float iou_threshold);
    friend Network& operator<<(Network &net, const Layer &layer);
protected:
    std::string image_file_name_;
    std::string weights_file_name_;

    /// Input image shape
    Shape net_shape_;
    std::vector<float> input_image_tensor;

    /// Network layers
    std::vector<Layer*> layers_;
    std::vector<float> utility_memory;
    std::vector<float> *current_tensor;
    BoundingBoxes bounding_boxes_;

    int num_classes_;

    friend class Layer;
    friend class ActivationLayer;
    friend class BatchNormLayer;
    friend class BiasLayer;
    friend class ConvolutionLayer;
    friend class LeakyReluLayer;
    friend class MaxPollingLayer;
    friend class UpsampleLayer;
    friend class ConcatenationLayer;
    friend class ShortcutLayer;
    friend class YoloLayer;
};
#endif //NETWORK_H
