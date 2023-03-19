#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H

#include "layers/layer.h"
#include <vector>

class UpsampleLayer: public Layer {
public:
    UpsampleLayer(int stride): stride_(stride),
                               Layer(LayerParameters(LayerType::UPSAMPLE,false)) {}
    ~UpsampleLayer() override {}
    void forward(Network &net) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    UpsampleLayer* clone() const override;
    int compute_out_width() const;
    int compute_out_height() const;
    int load_pretrained(std::ifstream &weights_file) override;
protected:
    int stride_;
    /// Output feature maps
    std::vector<float> outputs_;
};

#endif //UPSAMPLE_LAYER_H
