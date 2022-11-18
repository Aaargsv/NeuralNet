#ifndef MAX_POOLING_LAYER_H
#define CNN_MAX_POOLING_LAYER_H

#include "layer.h"

class MaxPollingLayer: public Layer {
private:
    void load_weights(std::ifstream &weights_file_input) override {};
public:
    /*pooling parameters*/
    int window_size_;
    int padding_;
    int stride_;

    MaxPollingLayer(int window_size, int padding, int stride): window_size_(window_size), padding_(padding),
                                                               stride_(stride), Layer(LayerType::MAX_POOLING, false) {}

    MaxPollingLayer(const MaxPollingLayer &max_pooling_layer): Layer(LayerType::MAX_POOLING, false) {
        window_size_ = max_pooling_layer.window_size_;
        padding_ = max_pooling_layer.padding_;
        stride_ = max_pooling_layer.stride_;
    }

    void forward(const std::vector<float> &input) override;
    void setup(int h, int w, int c) override;
    int compute_out_width();
    int compute_out_height();
    MaxPollingLayer* clone() const override;
    ~MaxPollingLayer() override;

};

#endif //MAX_POOLING_LAYER_H
