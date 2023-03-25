#ifndef NEURAL_H
#define NEURAL_H
#include <iostream>
#include <vector>

class Shape {
public:
    int h;
    int w;
    int c;

    Shape(): h(0), w(0), c(0) {}
    Shape(int h, int w, int c)
    {
        reshape(h, w, c);
    }

    void reshape(int h, int w, int c)
    {
        this->h = h;
        this->w = w;
        this->c = c;
    }

    int get_size()
    {
        return h * w * c;
    }

};

std::ostream& operator<<(std::ostream& os, const Shape &shape)
{
    os << "(" << shape.h << ", " << shape.w << ", " << shape.c << ")";
    return os;
}

bool is_HxW_equal(const Shape &shape1, const Shape &shape2)
{
    return ((shape1.h == shape2.h) && (shape1.w == shape2.w));
}

bool is_tensor_sizes_equal(const Shape &shape1, const Shape &shape2)
{
    return ((shape1.h == shape2.h) && (shape1.w == shape2.w) &&
            (shape1.c == shape2.c));
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
    LOGISTIC,
    YOLO
};

class LayerParameters {
public:
    bool can_pretrained_be_loaded_;
    LayerType layer_type_;
    LayerParameters(LayerType layer_type, bool can_pretrained_be_loaded):
                    layer_type_(layer_type),
                    can_pretrained_be_loaded_(can_pretrained_be_loaded) {}

};

struct BoundingBox  {
    float bx;
    float by;
    float bh;
    float bw;
    float probability;
};

typedef std::vector<std::vector<BoundingBox>> BoundingBoxes;


#endif //NEURAL_H
