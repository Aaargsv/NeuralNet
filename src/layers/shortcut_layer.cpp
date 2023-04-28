#include "layers/shortcut_layer.h"
#include "network.h"
#include "operations/tensor_math.h"


void ShortcutLayer::forward(Network &net)
{
    std::vector<Layer*> layers = net.layers_;
    int size = out_shape_.get_size();
    add_tensors(layers[prev_layer_index_]->get_outputs(),
                layers[index_]->get_outputs(),
                size, outputs_);
    print_info();
}

int ShortcutLayer::setup(const Shape &shape, const Network &net)
{
    Layer *layer0 = net.layers_[prev_layer_index_];
    Layer *layer1 = net.layers_[index_];
    if (!is_tensor_sizes_equal(layer0->out_shape(), layer1->out_shape())) {
        return -1;
    }
    out_shape_ = layer0->out_shape();
    outputs_.resize(out_shape_.get_size());
    return 0;
}

const std::vector<float> &ShortcutLayer::get_outputs()
{
    return outputs_;
}

void ShortcutLayer::print_info() const
{

}

ShortcutLayer *ShortcutLayer::clone() const
{
    return new ShortcutLayer(*this);
}

int ShortcutLayer::load_pretrained(std::ifstream &weights_file)
{
    return 0;
};
