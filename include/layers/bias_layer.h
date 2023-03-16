#ifndef BIAS_LAYER_H
#define BIAS_LAYER_H

#include "layers/layer.h"
#include "operations/tensor_math.h"

class BiasLayer: public Layer {
public:
    BiasLayer(): Layer(LayerParameters(LayerType::BIAS,true)) {}
    ~BiasLayer() override {};
    void forward(Network &net) override;
    int setup(const Shape &shape);
    int load_pretrained(std::ifstream &weights_file) override;
    void print_info() const override;
    BiasLayer* clone() const override;
protected:
    std::vector<float> bias_;
};
#endif //CNN_BIAS_LAYER_H
