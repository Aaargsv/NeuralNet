#include "layers/concatenation_layer.h"
#include "network.h"


void ConcatenationLayer::forward(Network &net)
{
    std::vector<Layer*> layers = net.layers_;
    for (int i = 0; i < indexes_.size(); i++) {
        outputs_.insert(outputs_.end(),
                        layers[indexes_[i]]->get_outputs().begin(),
                        layers[indexes_[i]]->get_outputs().end());
    }
    print_info();
}

int ConcatenationLayer::setup(const Shape &shape, const Network &net)
{
    std::vector<Layer*> layers = net.layers_;
    Shape out_shape = layers[indexes_[0]]->out_shape();
    for (int i = 1; i < indexes_.size(); i++) {
        Shape tmp_shape = layers[indexes_[i]]->out_shape();
        if(!is_HxW_equal(out_shape, tmp_shape)) {
            return -1;
        }
        out_shape.c_ += tmp_shape.c_;
    }

    out_shape_ = out_shape;
    outputs_.reserve(out_shape_.get_size());
    return 0;
}

const std::vector<float> &ConcatenationLayer::get_outputs()
{
    return outputs_;
}

void ConcatenationLayer::print_info() const
{

}

ConcatenationLayer *ConcatenationLayer::clone() const
{
    return new ConcatenationLayer(*this);
}
int ConcatenationLayer::compute_out_width() const
{
    return (in_shape_.w_ + padding_ - window_size_) / stride_ + 1;
}
int ConcatenationLayer::compute_out_height() const
{
    return (in_shape_.h_ + padding_ - window_size_) / stride_ + 1;
}

int ConcatenationLayer::load_pretrained(std::ifstream &weights_file)
{
    return 0;
};
