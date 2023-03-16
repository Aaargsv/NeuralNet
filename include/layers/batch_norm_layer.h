#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include "neural.h"
#include "layer.h"
#include <vector>

class BatchNormLayer: public Layer {
public:
    BatchNormLayer(): Layer(LayerParameters(LayerType::BATCH_NORM,true)) {}
    ~BatchNormLayer() override {};
    void forward(Network &net)  override;
    int setup(const Shape &shape) override;
    int load_pretrained(std::ifstream &input_file) override;
    void print_info() const override;
    BatchNormLayer* clone() const override;
protected:
    std::vector<float> rolling_mean_;
    std::vector<float> rolling_variance_;
    std::vector<float> gamma_;
    std::vector<float> beta_;
};

#endif //BATCH_NORM_LAYER_H
