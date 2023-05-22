#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "layers/layer.h"
#include "bounding_box.h"
#include <vector>

class YoloLayer: public Layer {
public:
    YoloLayer(int number_of_anchor_boxes, int classes,
              float threshold, std::vector<float> anchors):
        number_of_anchor_boxes_(number_of_anchor_boxes),
        classes_(classes), threshold_(threshold), anchors_(anchors),
        Layer(LayerParameters(LayerType::YOLO,false)) {}

    ~YoloLayer() override {}
    void forward(Network &net) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    YoloLayer* clone() const override;
    int load_pretrained(std::ifstream &weights_file, std::ofstream &check_file) override;
    int get_element_pos(int number_of_anchor, int component_index, int cell_h, int cell_w) const;
    void get_bounding_boxes(BoundingBoxes &bounding_boxes, int net_height, int net_width) const;
    int get_number_detection() const;
protected:
    int number_of_anchor_boxes_;
    int classes_;
    float threshold_;
    std::vector<float> anchors_;
    /// Output feature maps
    std::vector<float> outputs_;
};

#endif //YOLO_LAYER_H
