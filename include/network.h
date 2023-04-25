#ifndef NETWORK_H
#define NETWORK_H

#include "neural.h"
#include "layers/layer.h"
#include "bounding_box.h"
#include <vector>
#include <string>

class Network {
public:
    Network(int height, int width,
            const std::string &image_file_name,
            const std::string &weight_file_name,
            int &error_status);
    ~Network();

    inline void add_layer(Layer *layer) {
        layers_.push_back(layer);
    }
    int infer();
    std::vector<float>* forward(std::vector<float> *input_image);
    int setup();
    int load_image(const std::string &filename);
    void bilinear_interpolation(std::vector<float> &src, int src_height, int src_width,
            std::vector<float> &dst, int dst_height, int dst_width, int channels);
    int load_pretrained(const std::string &filename);
    void gather_bounding_boxes();
    void apply_nms(float iou_threshold);
    void draw_box(int x1, int y1, int x2, int y2, float r, float g, float b);
    void trace_box_outline(int x1, int y1, int x2, int y2, int w, float r, float g, float b);
    void draw_bounding_boxes(float threshold);
    int save_image(const std::string &file_name);

    friend Network& operator<<(Network &net, const Layer &layer);
protected:
    std::string image_file_name_;
    std::string weights_file_name_;

    /// Resized image shape
    Shape net_shape_;
    /// Input imahe shape
    Shape input_image_shape_;
    std::vector<float> input_image_tensor;
    std::vector<float> resized_image_tensor;

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
