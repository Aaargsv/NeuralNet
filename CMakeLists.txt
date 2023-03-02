cmake_minimum_required(VERSION 3.17)
project(CNN)

set(CMAKE_CXX_STANDARD 14)
include_directories(include)
add_executable(CNN main.cpp include/layers/layer.h src/network.cpp include/layers/network.h include/neural.h include/layers/activation_layer.h include/layers/leaky_relu_layer.h include/layers/convolution_layer.h src/layers/convolution_layer.cpp include/layers/max_pooling_layer.h src/layers/max_pooling_layer.cpp include/layers/batch_norm_layer.h src/layers/batch_norm_layer.cpp include/utils/utils.h src/utils/utils.cpp include/layers/bias_layer.h include/algorithms/im2col.h include/algorithms/gemm.h include/algorithms/convolution.h)