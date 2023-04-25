#include "layers/concatenation_layer.h"
#include "network.h"
#include "operations/tensor_math.h"


void ConcatenationLayer::forward(Network &net)
{
    std::vector<Layer*> layers = net.layers_;
    for (int i = 0; i < indexes_.size(); i++) {
        concatenate(outputs_, layers[indexes_[i]]->get_outputs());
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
            std::cout << "[Error]: can't concatenate" << std::endl;
            std::cout << "HxW isn't equal: " << "(" << out_shape.h << ", " <<  out_shape.w << ") "
                      << "and " <<  "(" << tmp_shape.h << ", " <<  tmp_shape.w << ")" << std::endl;
            return -1;
        }
        out_shape.c += tmp_shape.c;
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

int ConcatenationLayer::load_pretrained(std::ifstream &weights_file)
{
    return 0;
};
