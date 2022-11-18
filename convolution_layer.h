#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "layer.h"
#include "neural.h"

class ConvolutionLayer: public Layer {
private:
    //
public:
    /*convolution parameters*/
    int kernel_size_;
    int filters_;
    int padding_;
    int stride_;

    ConvolutionLayer(int kernel_size,
                     int filters,
                     int padding,
                     int stride): kernel_size_(kernel_size), filters_(filters),
                                  padding_(padding), stride_(stride),
                                  Layer(LayerType::CONVOLUTION, true) {}

    ConvolutionLayer(const ConvolutionLayer &conv_layer);
    void forward(const std::vector<float> &input) override;
    void load_weights(std::ifstream &weights_file_input) override;
    void setup(int h, int w, int c) override;
    int compute_out_width();
    int compute_out_height();

    ConvolutionLayer* clone() const override {
        return new ConvolutionLayer(*this);
    }

    ~ConvolutionLayer() override;

};

#endif //CONVOLUTION_LAYER_H
