#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"
#include "neural.h"

class ActivationLayer: public Layer {
public:
    ActivationLayer(const LayerParameters &layer_param): Layer(layer_param) {}
    ~ActivationLayer() {}
    void forward(Network &net) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;


protected:
    virtual float activation(float x) const = 0;
    std::vector<float> *outputs_ptr_;
private:
    int load_pretrained(std::ifstream &weights_file, std::ofstream &check_file) override;
};

#endif //ACTIVATION_LAYER_H
