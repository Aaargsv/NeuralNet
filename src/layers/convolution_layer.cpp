#include "layers/convolution_layer.h"
#include "operations/convolution.h"
#include "utils/utils.h"
#include "network.h"
#include "bounding_box.h"
#include <iostream>

void ConvolutionLayer::forward(Network &net)
{
    std::vector<float> &input = *net.current_tensor;

    convolution(input, in_shape_.c, in_shape_.h, in_shape_.w,
                kernel_size_, stride_, padding_, weights_, out_shape_.c,
                net.utility_memory, out_shape_.h, out_shape_.w, outputs_);

    net.current_tensor = &outputs_;
    inter_layer->forward(net);
    print_info();
    inter_layer->print_info();
}

int ConvolutionLayer::setup(const Shape &shape, const Network &net)
{
    in_shape_ = shape;
    /// channels * k * k * filters
    weights_length_ = in_shape_.c * kernel_size_ * kernel_size_ * filters_;
    out_shape_.reshape(compute_out_height(), compute_out_width(), filters_);


    std::cout << "[Convolution] " << "(" << filters_ << ", "
                                  << kernel_size_ << ", " << padding_ << ", " << stride_  << ") :"
                                  << out_shape_.h << std::endl;

    weights_.reserve(weights_length_);
    outputs_.reserve(out_shape_.get_size());
    if (has_batch_norm_)
        inter_layer = new BatchNormLayer();
    else
        inter_layer = new BiasLayer();
    inter_layer->setup(out_shape_, net);
    int utility_memory_size = out_shape_.h * out_shape_.w * kernel_size_ *
            kernel_size_ * out_shape_.c;
    return utility_memory_size;
}

const std::vector<float> &ConvolutionLayer::get_outputs()
{
    return outputs_;
}

int ConvolutionLayer::load_pretrained(std::ifstream &weights_file)
{

    if ( inter_layer->load_pretrained(weights_file))
        return 1;
    if(!weights_file.read(
            reinterpret_cast<char*>(weights_.data()),
              weights_length_ * sizeof(float)))
        return 1;
    std::cout << "convolution load weights\n";
    return 0;
}

void ConvolutionLayer::print_info() const
{
    std::cout << "LAYER NAME: CONVOLUTION\n";
    std::cout << "KERNEL SIZE: "    << kernel_size_ << "\n";
    std::cout << "FILTERS: "        << filters_ << "\n";
    std::cout << "PADDING: "        << padding_ << "\n";
    std::cout << "STRIDE: "         << stride_ << "\n";
    std::cout << "INPUT TENSOR: "   << in_shape_ << "\n";
    std::cout << "OUTPUT TENSOR: "  << out_shape_ << "\n";
    std::cout << "------------------------\n";
}

int ConvolutionLayer::compute_out_height() const
{
    return (in_shape_.h + 2 * padding_ - kernel_size_) / stride_ + 1;
}

int ConvolutionLayer::compute_out_width() const
{
    return (in_shape_.w + 2 * padding_ - kernel_size_) / stride_ + 1;
}

ConvolutionLayer *ConvolutionLayer::clone() const
{
    return new ConvolutionLayer(*this);
}