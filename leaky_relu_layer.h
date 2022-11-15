#ifndef LEAKY_RELU_LAYER_H
#define LEAKY_RELU_LAYER_H

class Leaky_ReLU_layer: public activation_layer {
private:
    //
public:
    void forward(const std::vector<float>& input) override {
        output_ = &input;
        for (int i = 0; i < input.size(); i++) {
            (*output_)[i] = activation(input[i]);
        }
    }
    
    float activation(float x) const override {
        return (x > 0) : x : 0.1 * x;
    }

    Leaky_ReLU_layer* clone() const override {
        return new Leaky_ReLU_layer();
    }
};
#endif //LEAKY_RELU_LAYER_H
