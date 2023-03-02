#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "layer.h"
#include "neural.h"
#include "batch_norm_layer.h"
#include "bias_layer.h"
#include <iostream>

class ConvolutionLayer: public Layer {
public:
    ConvolutionLayer(int kernel_size,
                     int filters,
                     int padding,
                     int stride,
                     bool has_batch_norm=false):
                     kernel_size_(kernel_size), filters_(filters), inter_layer(nullptr),
                     padding_(padding), stride_(stride), has_batch_norm_(has_batch_norm),
                     Layer(LayerParameters(LayerType::CONVOLUTION,true)) {}
    ~ConvolutionLayer() override {
       delete inter_layer;
    }

    void forward(std::vector<float> *input_tensor,
                 std::vector<float> *output_tensor) override;
    int load_pretrained(std::ifstream &input_file) override;
    void setup(const Shape &shape) override;
    void print_info() const override;

    inline int compute_out_height() const {
        return (in_shape_.h_ + 2 * padding_ - kernel_size_) / stride_ + 1;
    }

    inline int compute_out_width() const {
        return (in_shape_.w_ + 2 * padding_ - kernel_size_) / stride_ + 1;
    }

    inline ConvolutionLayer* clone() const override {
        return new ConvolutionLayer(*this);
    }

protected:
    /// False - Bias. True - Batch_norm.
    bool has_batch_norm_;
    /// Convolution parameters.
    int kernel_size_;
    int filters_;
    int padding_;
    int stride_;
    int weights_length_;
    /// Bias layer or batch_norm layer.
    Layer *inter_layer;
    std::vector<float> weights_;
    std::vector<float> outputs_;
};

#endif //CONVOLUTION_LAYER_H
