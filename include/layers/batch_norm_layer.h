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
    int setup(const Shape &shape, const Network &net) override;
    int load_pretrained(std::ifstream &weights_file, std::ofstream &check_file) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    BatchNormLayer* clone() const override;

protected:
    std::vector<float> rolling_mean_;
    std::vector<float> rolling_variance_;
    std::vector<float> gamma_;
    std::vector<float> beta_;
    std::vector<float> *ouputs_ptr;

};

#endif //BATCH_NORM_LAYER_H
