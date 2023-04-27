#include "layers/gpu/convolution_layer_gpu.h"
#include "operations/convolution.h"
#include "utils/utils.h"
#include "network.h"
#include "bounding_box.h"
#include "gpu.cuh"
#include <iostream>
#include <assert.h>

void ConvolutionLayerGPU::forward(Network &net)
{
    std::vector<float> &input = *net.current_tensor;

    float *dev_input;
    int error_status = gpu_malloc(dev_input, input.size());
    assert(error_status == 0);
    error_status = copy_to_gpu(dev_input, &input[0], input.size());
    assert(error_status == 0);

    convolution_gpu(dev_input, in_shape_.c, in_shape_.h, in_shape_.w,
                kernel_size_, stride_, padding_, dev_weights_, out_shape_.c,
                net.dev_utility_memory, out_shape_.h, out_shape_.w, dev_outputs_);

    error_status = extract_from_gpu(&outputs_[0], dev_outputs_, outputs_.size());
    assert(error_status == 0);

    net.current_tensor = &outputs_;
    inter_layer->forward(net);
    print_info();
    inter_layer->print_info();
}

int ConvolutionLayerGPU::setup(const Shape &shape, const Network &net)
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

    if (gpu_malloc(dev_weights_, weights_length_)) {
        std::cout << "[Error]: can't setup weights of ConvolutionLayerGPU" << std::endl;
        return -1;
    }

    if (gpu_malloc(dev_outputs_, weights_length_)) {
        std::cout << "[Error]: can't setup weights of ConvolutionLayerGPU" << std::endl;
        return -1;
    }

    if (has_batch_norm_)
        inter_layer = new BatchNormLayer();
    else
        inter_layer = new BiasLayer();
    inter_layer->setup(out_shape_, net);
    int utility_memory_size = out_shape_.h * out_shape_.w * kernel_size_ *
                              kernel_size_ * out_shape_.c;
    return utility_memory_size;
}


int ConvolutionLayerGPU::load_pretrained(std::ifstream &weights_file)
{

    if ( inter_layer->load_pretrained(weights_file))
        return 1;
    if(!weights_file.read(
            reinterpret_cast<char*>(weights_.data()),
            weights_length_ * sizeof(float)))
        return 1;

    if (copy_to_gpu(dev_weights_, &weights_[0], weights_.size())) {
        std::cout << "[Error]: can't copy weights of ConvolutionLayerGPU to GPU" << std::endl;
        return 1;
    }

    std::cout << "convolution load weights\n";
    return 0;
}


ConvolutionLayerGPU *ConvolutionLayerGPU::clone() const
{
    return new ConvolutionLayerGPU(*this);
}