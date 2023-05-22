#include "layers/upsample_layer.h"
#include "operations/upsampling.h"
#include "network.h"

void UpsampleLayer::forward(Network &net)
{
    std::vector<float> &input = *net.current_tensor;
    upsample(input, in_shape_.c, in_shape_.h, in_shape_.w, stride_, outputs_);
    net.current_tensor = &outputs_;
}

int UpsampleLayer::setup(const Shape &shape, const Network &net)
{
    in_shape_ = shape;
    out_shape_.reshape(compute_out_height(), compute_out_width(), shape.c);
    outputs_.resize(out_shape_.get_size());
    return 0;
}

const std::vector<float> &UpsampleLayer::get_outputs()
{
    return outputs_;
}

void UpsampleLayer::print_info() const
{


}

UpsampleLayer *UpsampleLayer::clone() const
{
    return new UpsampleLayer(*this);
}

int UpsampleLayer::compute_out_width() const
{
    return in_shape_.w * stride_;
}

int UpsampleLayer::compute_out_height() const
{
    return in_shape_.h * stride_;
}

int UpsampleLayer::load_pretrained(std::ifstream &weights_file, std::ofstream &check_file)
{
    return 0;
};



