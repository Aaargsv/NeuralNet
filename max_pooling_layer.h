#ifndef MAX_POOLING_LAYER_H
#define CNN_MAX_POOLING_LAYER_H

#include "layer.h"

class MaxPollingLayer: public Layer {
public:
    MaxPollingLayer(int window_size, int padding, int stride): window_size_(window_size),
                                                               padding_(padding),
                                                               stride_(stride),
                                                               Layer(LayerParameters(LayerType::MAX_POOLING,false)) {}
    ~MaxPollingLayer() override {};

    void forward(std::vector<float> *input_tensor,
                 std::vector<float> *output_tensor) override;
    void setup(const Shape &shape) override;
    void print_info() const override;

    inline MaxPollingLayer* clone() const override {
        return new MaxPollingLayer(*this);
    }
    inline int compute_out_width() {
        return (in_shape_.w_ + padding_ - window_size_) / stride_ + 1;
    }
    inline int compute_out_height() {
        return (in_shape_.h_ + padding_ - window_size_) / stride_ + 1;
    }


protected:
    /// Pooling parameters
    int window_size_;
    int padding_;
    int stride_;
    /// Output feature maps
    std::vector<float> outputs_;

private:
    void load_pretrained(std::ifstream &input_file) override {};
};

#endif //MAX_POOLING_LAYER_H
