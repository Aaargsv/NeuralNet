#include "network.h"
#include "layers/convolution_layer.h"
#include "layers/gpu/convolution_layer_gpu.h"
#include "layers/max_pooling_layer.h"
#include "layers/leaky_relu_layer.h"
#include "layers/linear_layer.h"
#include "layers/yolo_layer.h"
#include "layers/concatenation_layer.h"
#include "layers/upsample_layer.h"
#include "gpu.cuh"
#include <iostream>
#include <vector>
#include <string>

int main()
{
#ifdef GPU
    std::cout << "GPU MODE" << std::endl;
#endif

    std::string  image_file = "P0294.png";
    std::string  weight_file = "yolov3-tiny_obj_best.weights";

    display_header();

    int error_status = 0;
    Network model(416, 416, image_file, weight_file, 1, error_status);
    if (error_status) {
        std::cout << "[Error]: cnn" << std::endl;
        return 1;
    }
    float threshold = 0.0;

    model << /*0*/  ConvolutionLayerGPU(3, 16, 1, 1, true)
          << /*1*/  LeakyReluLayer()
          << /*2*/  MaxPollingLayer(2, 1, 2) /// padding is zero???
          << /*3*/  ConvolutionLayerGPU(3, 32, 1, 1, true)
          << /*4*/  LeakyReluLayer()
          << /*5*/  MaxPollingLayer(2, 1, 2)
          << /*6*/  ConvolutionLayerGPU(3, 64, 1, 1, true)
          << /*7*/  LeakyReluLayer()
          << /*8*/  MaxPollingLayer(2, 1, 2)
          << /*9*/  ConvolutionLayerGPU(3, 128, 1, 1, true)
          << /*10*/ LeakyReluLayer()
          << /*11*/ MaxPollingLayer(2, 1, 2)
          << /*12*/ ConvolutionLayerGPU(3, 256, 1, 1, true)
          << /*13*/ LeakyReluLayer()
          << /*14*/ MaxPollingLayer(2, 1, 2)
          << /*15*/ ConvolutionLayerGPU(3, 512, 1, 1, true)
          << /*16*/ LeakyReluLayer()
          << /*17*/ MaxPollingLayer(2, 1, 1) /// stride is 1???
          << /*18*/ ConvolutionLayerGPU(3, 1024, 1, 1, true)
          << /*19*/ LeakyReluLayer()
          /**********************************************/
          << /*20*/ ConvolutionLayerGPU(1, 256, 0, 1, true)
          << /*21*/ LeakyReluLayer()
          << /*22*/ ConvolutionLayerGPU(3, 512, 1, 1, true)
          << /*23*/ LeakyReluLayer()
          << /*24*/ ConvolutionLayerGPU(1, 18, 0, 1, false)
          << /*25*/ LinearLayer()
          << /*26*/ YoloLayer(3, 1, threshold, {81,82,  135,169,  344,319})
          << /*27*/ ConcatenationLayer({20})
          << /*28*/ ConvolutionLayerGPU(1, 128, 0, 1, true)
          << /*29*/ LeakyReluLayer()
          << /*30*/ UpsampleLayer(2)
          << /*31*/ ConcatenationLayer({30, 11})
          << /*32*/ ConvolutionLayerGPU(3, 256, 1, 1, true)
          << /*33*/ LeakyReluLayer()
          << /*34*/ ConvolutionLayerGPU(1, 18, 0, 1, false)
          << /*35*/ LinearLayer()
          << /*36*/ YoloLayer(3, 1, threshold, {23,27,  37,58,  81,82});

    if (model.infer()) {
        std::cout << "[Error]: can't infer" << std::endl;
        return 1;
    }

    model.save_detections("log_detections.txt");

    //model.apply_nms(0.4);
    model.draw_bounding_boxes(0.28);
    if (model.save_image("detection")) {
        std::cout << "[Error]: can't save file" << std::endl;
        return 1;
    }

    /**
     *  Network cnn(416, 416, 3);
     *  cnn << ConvolutionLayerGPU(3, 16, 1, 1, true)
     *  << MaxPollingLayer(2, 2, 2)
     *  << ConvolutionLayerGPU(3, 32, 1, 1, false);
     *  cnn.setup();

     *  std::vector<float> image(416 * 416 * 3);
     *  cnn.forward(&image);
     */

    return 0;
}
