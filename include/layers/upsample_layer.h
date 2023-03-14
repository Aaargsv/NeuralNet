#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H

#include "layers/layer.h"
#include <vector>

class UpsampleLayer: public Layer {
public:
    UpsampleLayer(int stride): stride_(stride) {}
    ~UpsampleLayer() override {}
    std::vector<float> *forward(std::vector<float> *input_tensor,
                                std::vector<float> &utility_memory) override;
    int setup(const Shape &shape) override;
    void print_info() const override;
    
    inline UpsampleLayer* clone() const override {
        return new UpsampleLayer(*this);
    }
    inline int compute_out_width() {
        return in_shape_.w_ * stride_;
    }
    inline int compute_out_height() {
        return in_shape_.h_ * stride_;
    }
protected:
    int stride_;
    /// Output feature maps
    std::vector<float> outputs_;

};

#endif //UPSAMPLE_LAYER_H
