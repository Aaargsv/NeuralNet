#ifndef LAYER_H
#define LAYER_H

#include "neural.h"
#include "nnfwd.h"
#include <vector>
#include <fstream>

class Layer {
public:
    Layer(const LayerParameters &layer_param): layer_param_(layer_param) {}
    virtual ~Layer() {};

    virtual void forward(Network &net) = 0;

    /**
    * @brief Initialize layer in_shape, compute out_shape,
    *        allocate resources.
    * @return utility memory size
    */
    virtual int setup(const Shape &shape) = 0;
    virtual int load_pretrained(std::ifstream &weights_file) = 0;
    virtual Layer *clone() const = 0;
    virtual void print_info() const = 0;
    virtual int get_utility_memory_size() const {
        return 0;
    };

    inline const LayerParameters &layer_param() const {
        return layer_param_;
    }
    inline const Shape &in_shape() const {
        return in_shape_;
    }
    inline const Shape &out_shape() const {
        return out_shape_;
    }

protected:
    LayerParameters layer_param_;
    /// Input and output tensor shapes
    Shape in_shape_;
    Shape out_shape_;
};
#endif //LAYER_H
