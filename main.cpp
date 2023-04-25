#include "network.h"
#include "layers/convolution_layer.h"
#include "layers/max_pooling_layer.h"
#include "layers/leaky_relu_layer.h"
#include "layers/linear_layer.h"
#include "layers/yolo_layer.h"
#include "layers/concatenation_layer.h"
#include "layers/upsample_layer.h"
#include <iostream>
#include <vector>
#include <string>


int main() {

    std::string  image_file = "airport.png";
    std::string  weight_file = "yolov3-tiny_obj_best.weights";
    int error_status = 0;
    Network model(416, 416, image_file, weight_file, error_status);
    if (error_status) {
        std::cout << "[Error]: cnn" << std::endl;
        return 1;
    }

    model << /*0*/  ConvolutionLayer(3, 16, 1, 1, true)
          << /*1*/  LeakyReluLayer()
          << /*2*/  MaxPollingLayer(2, 1, 2) /// padding is zero???
          << /*3*/  ConvolutionLayer(3, 32, 1, 1, true)
          << /*4*/  LeakyReluLayer()
          << /*5*/  MaxPollingLayer(2, 1, 2)
          << /*6*/  ConvolutionLayer(3, 64, 1, 1, true)
          << /*7*/  LeakyReluLayer()
          << /*8*/  MaxPollingLayer(2, 1, 2)
          << /*9*/  ConvolutionLayer(3, 128, 1, 1, true)
          << /*10*/ LeakyReluLayer()
          << /*11*/ MaxPollingLayer(2, 1, 2)
          << /*12*/ ConvolutionLayer(3, 256, 1, 1, true)
          << /*13*/ LeakyReluLayer()
          << /*14*/ MaxPollingLayer(2, 1, 2)
          << /*15*/ ConvolutionLayer(3, 512, 1, 1, true)
          << /*16*/ LeakyReluLayer()
          << /*17*/ MaxPollingLayer(2, 1, 1) /// stride is 1???
          << /*18*/ ConvolutionLayer(3, 1024, 1, 1, true)
          << /*19*/ LeakyReluLayer()
          /**********************************************/
          << /*20*/ ConvolutionLayer(1, 256, 0, 1, true)
          << /*21*/ LeakyReluLayer()
          << /*22*/ ConvolutionLayer(3, 512, 1, 1, true)
          << /*23*/ LeakyReluLayer()
          << /*24*/ ConvolutionLayer(1, 18, 0, 1, false)
          << /*25*/ LinearLayer()
          << /*26*/ YoloLayer(6, 1, 0.7, {10,14,  23,27,  37,58,  81,82,  135,169,  344,319})
          << /*27*/ ConcatenationLayer({20})
          << /*28*/ ConvolutionLayer(1, 128, 0, 1, true)
          << /*29*/ LeakyReluLayer()
          << /*30*/ UpsampleLayer(2)
          << /*31*/ ConcatenationLayer({30, 11})
          << /*32*/ ConvolutionLayer(3, 256, 1, 1, true)
          << /*33*/ LeakyReluLayer()
          << /*34*/ ConvolutionLayer(1, 18, 0, 1, false)
          << /*35*/ LinearLayer()
          << /*36*/ YoloLayer(6, 1, 0.7, {10,14,  23,27,  37,58,  81,82,  135,169,  344,319});

    if (model.infer()) {
        std::cout << "[Error]: can't infer" << std::endl;
        return 1;
    }

    model.apply_nms(0.4);
    model.draw_bounding_boxes(0.4);
    if (model.save_image("detection")) {
        std::cout << "[Error]: can't save file" << std::endl;
        return 1;
    }

    /**
     *  Network cnn(416, 416, 3);
     *  cnn << ConvolutionLayer(3, 16, 1, 1, true)
     *  << MaxPollingLayer(2, 2, 2)
     *  << ConvolutionLayer(3, 32, 1, 1, false);
     *  cnn.setup();

     *  std::vector<float> image(416 * 416 * 3);
     *  cnn.forward(&image);
     */

    return 0;
}
