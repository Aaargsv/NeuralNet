#ifndef BIAS_LAYER_H
#define BIAS_LAYER_H
#include "layer.h"
class BiasLayer: public Layer {
public:
    BiasLayer(): Layer(LayerParameters(LayerType::BIAS,true)) {}
    ~BiasLayer() override {};

    inline void forward(std::vector<float> *input_tensor,
                 std::vector<float> *output_tensor)  override {
        output_tensor = input_tensor;
        std::vector<float> &input = *input_tensor;
        add_bias(input);
    }

    void add_bias(std::vector<float> &input) {
        int c = in_shape_.c_;
        int channel_size = in_shape_.h_ * in_shape_.w_;
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < channel_size; j++)
                input[i * channel_size + j] += bias_[i];
        }
    }

    inline void setup(const Shape &shape) {
        in_shape_ = shape;
        out_shape_ = shape;
        bias_.reserve(shape.c_);
    };
    void load_pretrained(std::ifstream &input_file) override {

    };

    inline void print_info() const override {
        std::cout << "LAYER NAME: BIAS\n";
        std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
        std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
        std::cout << "------------------------\n";
    }

    inline BiasLayer* clone() const override {
        return new BiasLayer(*this);
    };

protected:
    std::vector<float> bias_;
};
#endif //CNN_BIAS_LAYER_H
