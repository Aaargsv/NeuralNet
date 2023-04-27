#ifndef MAX_POOLING_LAYER_H
#define CNN_MAX_POOLING_LAYER_H

#include "layer.h"

class MaxPollingLayer: public Layer {
public:
    MaxPollingLayer(int window_size, int padding, int stride): window_size_(window_size),
                                                               padding_(padding),
                                                               stride_(stride),
                                                               Layer(LayerParameters(LayerType::MAX_POOLING,false)) {}
    ~MaxPollingLayer() override {};
    void forward(Network &net) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    MaxPollingLayer* clone() const override;
    int compute_out_width() const;
    int compute_out_height() const;
    int load_pretrained(std::ifstream &weights_file) override;

protected:
    /// Pooling parameters
    int window_size_;
    int padding_;
    int stride_;
    /// Output feature maps
    std::vector<float> outputs_;

private:

};

#endif //MAX_POOLING_LAYER_H
