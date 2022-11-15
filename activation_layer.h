#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

class ActivationLayer: public Layer {
private:
    void load_weights(std::ifstream& weights_file_input) override {}
public:
    virtual float activation(float x) const = 0;
};

#endif //ACTIVATION_LAYER_H
