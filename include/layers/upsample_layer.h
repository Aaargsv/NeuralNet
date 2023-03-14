#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H

#include "layers/layer.h"
#include <vector>

class UpsampleLayer: public Layer {
public:

protected:
    int stride;
    /// Output feature maps
    std::vector<float> outputs_;

};

#endif //UPSAMPLE_LAYER_H
