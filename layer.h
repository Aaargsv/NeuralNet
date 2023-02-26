#ifndef LAYER_H
#define LAYER_H

#include "neural.h"
#include <vector>
#include <fstream>

class Layer {
public:
    Layer(const LayerParameters &layer_param): layer_param_(layer_param) {}
    virtual ~Layer() {};

    virtual void forward(std::vector<float> *input_tensor,
                         std::vector<float> *output_tensor) = 0;

    /**
    * @brief Initialize layer in_shape, compute out_shape,
    *        allocate resources.
    */
    virtual void setup(const Shape &shape) = 0;
    virtual int load_pretrained(std::ifstream &input_file) = 0;
    virtual Layer* clone() const = 0;
    virtual void print_info() const = 0;

    inline const LayerParameters& layer_param() const {
        return layer_param_;
    }
    inline const Shape& in_shape() const {
        return in_shape_;
    }
    inline const Shape& out_shape() const {
        return out_shape_;
    }

protected:
    LayerParameters layer_param_;
    /// Input and output tensor shapes
    Shape in_shape_;
    Shape out_shape_;
};
#endif //LAYER_H
