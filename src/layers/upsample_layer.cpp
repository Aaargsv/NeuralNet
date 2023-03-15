#include "layers/upsample_layer.h"
#include "operations/upsampling.h"

std::vector<float> *UpsampleLayer::forward(std::vector<float> *input_tensor,
                                           std::vector<float> &utility_memory)
{
    std::vector<float> &input = *input_tensor;
    upsample(input, in_shape_.c_, in_shape_.h_, in_shape_.w_, stride_, outputs_);
    return &outputs_;
}

int UpsampleLayer::setup(const Shape &shape)
{
    in_shape_ = shape;
    out_shape_.reshape(compute_out_height(), compute_out_width(), shape.c_);
    outputs_.reserve(out_shape_.get_size());
    return 0;
}
void UpsampleLayer::print_info() const
{


}




