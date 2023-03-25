#ifndef CONCATENATION_LAYER_H
#define CONCATENATION_LAYER_H

#include "layer.h"

class ConcatenationLayer: public Layer {
public:
    ConcatenationLayer(std::vector<int> &indexes): indexes_(indexes),
            Layer(LayerParameters(LayerType::CONCATENATION,false)) {}
    ~ConcatenationLayer() override {}

    void forward(Network &net) override;
    int load_pretrained(std::ifstream &input_file) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    ConcatenationLayer* clone() const override;

protected:
    std::vector<int> indexes_;
    std::vector<float> outputs_;

};

#endif //CONCATENATION_LAYER_H
