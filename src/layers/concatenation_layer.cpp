#include "layers/concatenation_layer.h"
#include "network.h"
#include "operations/tensor_math.h"


void ConcatenationLayer::forward(Network &net)
{
    std::vector<Layer*> layers = net.layers_;
    int start = 0;
    for (int i = 0; i < indexes_.size(); i++) {
        const std::vector<float> &src = layers[indexes_[i]]->get_outputs();
        int len = src.size();
        copy_vector(outputs_, src, start, len);
        start += len;
        //concatenate(outputs_, layers[indexes_[i]]->get_outputs());
    }
    net.current_tensor = &outputs_;
    //print_info();
}

int ConcatenationLayer::setup(const Shape &shape, const Network &net)
{
    std::vector<Layer*> layers = net.layers_;
    Shape out_shape = layers[indexes_[0]]->out_shape();
    std::cout << "[concat shapes]" << out_shape.c << " ";
    for (int i = 1; i < indexes_.size(); i++) {
        Shape tmp_shape = layers[indexes_[i]]->out_shape();
        std::cout << tmp_shape.c << " ";
        if(!is_HxW_equal(out_shape, tmp_shape)) {
            std::cout << "[Error]: can't concatenate" << std::endl;
            std::cout << "HxW isn't equal: " << "(" << out_shape.h << ", " <<  out_shape.w << ") "
                      << "and " <<  "(" << tmp_shape.h << ", " <<  tmp_shape.w << ")" << std::endl;
            return -1;
        }
        out_shape.c += tmp_shape.c;
    }
    std::cout << std::endl;
    out_shape_ = out_shape;

    outputs_.resize(out_shape_.get_size());
    std::cout << "ConcatenationLayer channels: " << out_shape_.c << std::endl;
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

int ConcatenationLayer::load_pretrained(std::ifstream &weights_file, std::ofstream &check_file)
{
    return 0;
};
