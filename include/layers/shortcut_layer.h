#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"

class ShortcutLayer: public Layer {
public:
    ShortcutLayer(int prev_layer_index, int index): prev_layer_index_(prev_layer_index),
        index_(index),
        Layer(LayerParameters(LayerType::SHORTCUT,false)) {}
    ~ShortcutLayer() override {}

    void forward(Network &net) override;
    int load_pretrained(std::ifstream &input_file) override;
    int setup(const Shape &shape, const Network &net) override;
    const std::vector<float> &get_outputs() override;
    void print_info() const override;
    int compute_out_height() const;
    int compute_out_width() const;
    ShortcutLayer* clone() const override;

protected:
    int index_;
    int prev_layer_index_;
    std::vector<float> outputs_;
};

#endif //HORTCUT_LAYER_H
