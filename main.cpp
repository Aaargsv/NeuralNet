#include "network.h"
#include "convolution_layer.h"
#include "max_pooling_layer.h"
#include "leaky_relu_layer.h"
#include <iostream>
#include <vector>


int main() {
    Network cnn(416, 416, 3);
    cnn << ConvolutionLayer(3, 16, 1, 1, true)
        << MaxPollingLayer(2, 2, 2)
        << ConvolutionLayer(3, 32, 1, 1, false)
  ;
    cnn.setup();


    std::vector<float> image(416 * 416 * 3);
    cnn.forward(&image);

    return 0;
}
