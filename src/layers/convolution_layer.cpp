#include "layers/convolution_layer.h"
#include "operations/convolution.h"
#include "utils/utils.h"
#include "network.h"
#include "bounding_box.h"
#include "timer.h"
#include <iostream>
#include <string>

void ConvolutionLayer::forward(Network &net)
{
    std::vector<float> &input = *net.current_tensor;

    std::string conv_algorithm = (convolution_type_ == ConvType::IM2COL) ? "im2col"
            : (convolution_type_ == ConvType::KN2ROW) ? "kn2row" : "winograd3x3";

    Timer timer("Conv" + std::to_string(kernel_size_)
                + "x" + std::to_string(kernel_size_) +
                ": " + std::to_string(in_shape_.h) + "x"
                     + std::to_string(in_shape_.w) + "x"
                     + std::to_string(in_shape_.c) + " --> "
                     + std::to_string(out_shape_.h) + "x"
                     + std::to_string(out_shape_.w) + "x"
                     + std::to_string(out_shape_.c) + " " + conv_algorithm
                );
    if (convolution_type_ == ConvType::IM2COL) {
        convolution(input, in_shape_.c, in_shape_.h, in_shape_.w,
                    kernel_size_, stride_, padding_, weights_, out_shape_.c,
                    net.utility_memory, out_shape_.h, out_shape_.w, outputs_);
    } else if (convolution_type_ == ConvType::WINOGRAD3x3) {
        winograd_convolution(input, in_shape_.c, in_shape_.h, in_shape_.w,
                             kernel_size_, stride_, padding_, weights_, out_shape_.c,
                             net.utility_memory, out_shape_.h, out_shape_.w, outputs_);
    } else if (convolution_type_ == ConvType::KN2ROW) {
        kn2row_convolution(input, in_shape_.c, in_shape_.h, in_shape_.w,
                           kernel_size_, stride_, padding_, weights_, out_shape_.c,
                           net.utility_memory, out_shape_.h, out_shape_.w, outputs_);
    } else
        std::cout << "[Error]: wrong convolution algorithm" << std::endl;

    timer.stop();
    net.current_tensor = &outputs_;
    inter_layer->forward(net);
    //print_info();
    //inter_layer->print_info();
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

    weights_.resize(weights_length_);
    outputs_.resize(out_shape_.get_size());
    if (has_batch_norm_)
        inter_layer = new BatchNormLayer();
    else
        inter_layer = new BiasLayer();
    inter_layer->setup(out_shape_, net);

    int utility_memory_size;
    if (convolution_type_ == ConvType::IM2COL) {
        utility_memory_size = out_shape_.h * out_shape_.w * kernel_size_ *
                              kernel_size_ * out_shape_.c;
    } else if (convolution_type_ == ConvType::WINOGRAD3x3) {
        int tile_h = (out_shape_.h + 1) / 2;
        int tile_w = (out_shape_.w + 1) / 2;
        int size_w = in_shape_.c * out_shape_.c;
        int size_s = in_shape_.c * tile_h * tile_w;
        int size_d = out_shape_.c * tile_h * tile_w;
        utility_memory_size = 16 * (size_w + size_s + size_d);
    } else if (convolution_type_ == ConvType::KN2ROW) {
        utility_memory_size = kernel_size_ * kernel_size_ * out_shape_.c * in_shape_.c + in_shape_.h * in_shape_.w * out_shape_.c;
    }
    return utility_memory_size;
}

const std::vector<float> &ConvolutionLayer::get_outputs()
{
    return outputs_;
}

int ConvolutionLayer::load_pretrained(std::ifstream &weights_file, std::ofstream &check_file)
{

    if ( inter_layer->load_pretrained(weights_file, check_file))
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