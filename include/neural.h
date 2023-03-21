#ifndef NEURAL_H
#define NEURAL_H
#include <iostream>

class Shape {
public:
    int h_;
    int w_;
    int c_;

    Shape(): h_(0), w_(0), c_(0) {}
    Shape(int h, int w, int c)
    {
        reshape(h, w, c);
    }

    void reshape(int h, int w, int c)
    {
        h_ = h;
        w_ = w;
        c_ = c;
    }

    int get_size()
    {
        return h_ * w_ * c_;
    }

};

std::ostream& operator<<(std::ostream& os, const Shape &shape)
{
    os << "(" << shape.h_ << ", " << shape.w_ << ", " << shape.c_ << ")";
    return os;
}

bool is_HxW_equal(const Shape &shape1, const Shape &shape2)
{
    return ((shape1.h_ == shape2.h_) && (shape1.w_ == shape2.w_));
}

bool is_tensor_sizes_equal(const Shape &shape1, const Shape &shape2)
{
    return ((shape1.h_ == shape2.h_) && (shape1.w_ == shape2.w_) &&
            (shape1.c_ == shape2.c_));
}

enum class LayerType {
    CONVOLUTION,
    MAX_POOLING,
    UPSAMPLE,
    CONCATENATION,
    SHORTCUT,
    BATCH_NORM,
    BIAS,
    LEAKY_ReLU,
    LOGISTIC
};

class LayerParameters {
public:
    bool can_pretrained_be_loaded_;
    LayerType layer_type_;
    LayerParameters(LayerType layer_type, bool can_pretrained_be_loaded):
                    layer_type_(layer_type),
                    can_pretrained_be_loaded_(can_pretrained_be_loaded) {}

};


#endif //NEURAL_H
