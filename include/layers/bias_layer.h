#ifndef BIAS_LAYER_H
#define BIAS_LAYER_H
#include "layers/layer.h"
#include "operations/tensor_math.h"

class BiasLayer: public Layer {
public:
    BiasLayer(): Layer(LayerParameters(LayerType::BIAS,true)) {}
    ~BiasLayer() override {};

    std::vector<float> *forward(std::vector<float> *input_tensor,
                                std::vector<float> &utility_memory)  override
    {
        std::vector<float> &input = *input_tensor;
        add_bias(input, bias_, in_shape_.c_, in_shape_.h_ * in_shape_.w_);
        return input_tensor;
    }

    inline int setup(const Shape &shape)
    {
        in_shape_ = shape;
        out_shape_ = shape;
        bias_.reserve(shape.c_);
        return 0;
    }

    int load_pretrained(std::ifstream &weights_file) override
    {
        if(!weights_file.read(reinterpret_cast<char*>(bias_.data()),
                              out_shape_.c_ * sizeof(float)))
            return 1;
        return 0;
    }

    inline void print_info() const override
    {
        std::cout << "LAYER NAME: BIAS\n";
        std::cout << "INPUT TENSOR: " << in_shape_ << "\n";
        std::cout << "OUTPUT TENSOR: " << out_shape_ << "\n";
        std::cout << "------------------------\n";
    }

    inline BiasLayer* clone() const override
    {
        return new BiasLayer(*this);
    };

protected:
    std::vector<float> bias_;
};
#endif //CNN_BIAS_LAYER_H
