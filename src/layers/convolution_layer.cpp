#include "layers/convolution_layer.h"
#include "operations/convolution.h"
#include "utils/utils.h"
#include <iostream>

std::vector<float> *ConvolutionLayer::forward(std::vector<float> *input_tensor,
                                              std::vector<float> &utility_memory) {

    std::vector<float> &input = *input_tensor;
    convolution(input, in_shape_.c_, in_shape_.h_, in_shape_.w_,
                kernel_size_, stride_, padding_, weights_, out_shape_.c_,
                utility_memory, out_shape_.h_, out_shape_.w_, outputs_);
    inter_layer->forward(&outputs_, utility_memory);
    print_info();
    inter_layer->print_info();
    return &outputs_;
}

int ConvolutionLayer::setup(const Shape &shape) {
    in_shape_ = shape;
    weights_length_ = kernel_size_ * kernel_size_ * filters_;
    out_shape_.reshape(compute_out_height(), compute_out_width(), filters_);
    weights_.reserve(weights_length_);
    outputs_.reserve(out_shape_.get_size());
    if (has_batch_norm_)
        inter_layer = new BatchNormLayer();
    else
        inter_layer = new BiasLayer();
    inter_layer->setup(out_shape_);
    int utility_memory_size = out_shape_.h_ * out_shape_.w_ * kernel_size_ *
            kernel_size_ * out_shape_.c_;
    return utility_memory_size;
}

int ConvolutionLayer::load_pretrained(std::ifstream &weights_file) {

    if ( inter_layer->load_pretrained(weights_file))
        return 1;
    if(!weights_file.read(
            reinterpret_cast<char*>(weights_.data()),
              weights_length_ * sizeof(float)))
        return 1;
    std::cout << "convolution load weights\n";
    return 0;
}

void ConvolutionLayer::print_info() const {
    std::cout << "LAYER NAME: CONVOLUTION\n";
    std::cout << "KERNEL SIZE: "    << kernel_size_ << "\n";
    std::cout << "FILTERS: "        << filters_ << "\n";
    std::cout << "PADDING: "        << padding_ << "\n";
    std::cout << "STRIDE: "         << stride_ << "\n";
    std::cout << "INPUT TENSOR: "   << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: "  << out_shape_ << "\n";
    std::cout << "------------------------\n";
}