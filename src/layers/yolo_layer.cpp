#include "layers/yolo_layer.h"
#include "operations/activation_functions.h"
#include "network.h"
#include "neural.h"
#include <cmath>
#include <vector>

void YoloLayer::forward(Network &net)
{
    outputs_ = *net.current_tensor;
    for (int n = 0; n < number_of_anchor_boxes_; n++) {
        int pos = get_element_pos(n, 0, 0);
        for (int i = 0; i < 2 * in_shape_.h * in_shape_.w; i++) {
            outputs_[pos + i] = logistic(outputs_[pos + i]);
        }
        pos = get_element_pos(n, 4, 0);
        for (int i = 0; i < (classes_ + 1) * in_shape_.h * in_shape_.w; i++) {
            outputs_[pos + i] = logistic(outputs_[pos + i]);
        }
    }
}

int YoloLayer::setup(const Shape &shape, const Network &net)
{
    in_shape_ = shape;
    out_shape_ = in_shape_;
    outputs_.reserve(out_shape_.get_size());
    return 0;
}

int YoloLayer::get_element_pos(int number_of_anchor, int component_index, int cell_h, int cell_w) const
{
    return number_of_anchor * in_shape_.h * in_shape_.w * (4 + classes_ + 1)
        + component_index * in_shape_.h * in_shape_.w + cell_h * in_shape_.w + cell_w;
}

int YoloLayer::get_number_detection() const
{
    int count = 0;
    for (int n = 0; n < number_of_anchor_boxes_; n++) {
        for (int h = 0; h < in_shape_.h; h++) {
            for (int w = 0; w < in_shape_.w; w++) {
                int pc_index = get_element_pos(n, 4, h, w);
                if (outputs_[pc_index] > threshold_) {
                    count++;
                }
            }
        }
    }
    return count;
}

void YoloLayer::get_bounding_boxes(BoundingBoxes &bounding_boxes, int net_height, int net_width) const
{
    for (int n = 0; n < number_of_anchor_boxes_; n++) {
        for (int h = 0; h < in_shape_.h; h++ ) {
            for (int w = 0; w < in_shape_.w; w++) {
                int confidence_index = get_element_pos(n, 4, h, w);
                if (confidence_index < threshold_)
                    continue;
                BoundingBox bb;
                int tx_index = get_element_pos(n, 0, h, w);
                int ty_index = get_element_pos(n, 1, h, w);
                int tw_index = get_element_pos(n, 2, h, w);
                int th_index = get_element_pos(n, 3, h, w);

                bb.bx = (outputs_[tx_index] + w) / in_shape_.w;
                bb.by = (outputs_[ty_index] + h) / in_shape_.h;
                bb.bw = std::exp(outputs_[tw_index]) * anchors_[2 * n] / net_width;
                bb.bh = std::exp(outputs_[th_index]) * anchors_[2 * n + 1] / net_height;

                for (int i = 0; i < classes_; i++) {
                    int class_index = get_element_pos(n, 5 + i, h, w);
                    float prob = outputs_[confidence_index] * outputs_[class_index];
                    bb.probability = prob;
                    if (bb.probability > threshold_) {
                        bounding_boxes[i].push_back(bb);
                    }
                }
            }
        }
    }
}

const std::vector<float> &YoloLayer::get_outputs()
{
    return outputs_;
}

void YoloLayer::print_info() const
{

}

YoloLayer *YoloLayer::clone() const
{
    return new YoloLayer(*this);
}


int YoloLayer::load_pretrained(std::ifstream &weights_file)
{
    return 0;
};



