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
                     bool has_batch_norm=false, ConvType convolution_type=ConvType::IM2COL):
                     kernel_size_(kernel_size), filters_(filters), inter_layer(nullptr),
                     padding_(padding), stride_(stride), has_batch_norm_(has_batch_norm), convolution_type_(convolution_type),
                     Layer(LayerParameters(LayerType::CONVOLUTION,true)) {}

    ~ConvolutionLayer() override {
       //delete inter_layer;
    }

    void forward(Network &net) override;
    int load_pretrained(std::ifstream &weights_file, std::ofstream &check_file) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    int compute_out_height() const;
    int compute_out_width() const;
    ConvolutionLayer* clone() const override;

protected:
    /// False - Bias. True - Batch_norm.
    bool has_batch_norm_;
    /// Convolution parameters.
    int kernel_size_;
    int filters_;
    int padding_;
    int stride_;
    int weights_length_;
    ConvType convolution_type_;
    /// Bias layer or batch_norm layer.
    Layer *inter_layer;
    std::vector<float> weights_;
    std::vector<float> outputs_;

};


#endif //CONVOLUTION_LAYER_H
