#ifndef CONVOLUTION_LAYER_GPU_H
#define CNN_CONVOLUTION_LAYER_GPU_H

#include "layers/convolution_layer.h"
#include "neural.h"
#include <iostream>

class ConvolutionLayerGPU: public ConvolutionLayer {
public:
    ConvolutionLayerGPU(int kernel_size,
                        int filters,
                        int padding,
                        int stride,
                        bool has_batch_norm = false) :
                        ConvolutionLayer(kernel_size, filters,  padding, stride, has_batch_norm) {}
    ~ConvolutionLayerGPU() override {
        delete inter_layer;
    }

    void forward(Network &net) override;
    int setup(const Shape &shape, const Network &net) override;
    int load_pretrained(std::ifstream &input_file) override;
    ConvolutionLayerGPU *clone() const override;

protected:
    float *dev_weights_;
    float *dev_outputs_;
};

#endif //CONVOLUTION_LAYER_GPU_H
