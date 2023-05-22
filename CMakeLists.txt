cmake_minimum_required(VERSION 3.17)
project(CNN LANGUAGES CXX CUDA)

find_package(CUDA)
include_directories(include)
SET(CMAKE_CUDA_STANDARD 14)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(CNN main.cpp include/layers/layer.h src/network.cpp include/network.h include/neural.h include/layers/activation_layer.h
        include/layers/leaky_relu_layer.h include/layers/convolution_layer.h src/layers/convolution_layer.cpp
        include/layers/max_pooling_layer.h src/layers/max_pooling_layer.cpp include/layers/batch_norm_layer.h
        src/layers/batch_norm_layer.cpp include/utils/utils.h src/utils/utils.cpp include/layers/bias_layer.h
        include/operations/im2col.h include/operations/gemm.h include/operations/convolution.h src/opeartions/im2col.cpp
        src/opeartions/gemm.cpp include/operations/tensor_math.h src/opeartions/tensor_math.cpp src/opeartions/convolution.cpp
        include/operations/max_pool.h src/opeartions/max_pool.cpp src/layers/upsample_layer.cpp include/layers/upsample_layer.h
        include/operations/upsampling.h src/opeartions/upsampling.cpp include/nnfwd.h src/layers/activation_layer.cpp
        src/layers/bias_layer.cpp src/layers/leaky_relu_layer.cpp include/layers/concatenation_layer.h
        src/layers/concatenation_layer.cpp include/layers/shortcut_layer.h src/layers/shortcut_layer.cpp
        include/layers/logistic_layer.h src/layers/logistic_layer.cpp include/layers/yolo_layer.h
        src/layers/yolo_layer.cpp include/operations/activation_functions.h src/opeartions/activation_functions.cpp
        include/stb_image.h include/stb_image_write.h include/layers/linear_layer.h src/layers/linear_layer.cpp
        include/bounding_box.h src/bounding_box.cpp
        include/layers/gpu/convolution_layer_gpu.h src/layers/gpu/convolution_layer_gpu.cpp
        include/layers/gpu/bias_layer_gpu.h src/layers/gpu/bias_layer_gpu.cpp
        include/layers/gpu/batch_norm_layer_gpu.h src/layers/gpu/batch_norm_layer_gpu.cpp

        src/gpu.cu
        include/gpu.cuh
        include/operations/im2col_gpu.cuh
        include/operations/gemm_gpu.cuh
        src/opeartions/gemm_gpu.cu
        src/opeartions/im2col.cu include/timer.h)

#add_compile_definitions(GPU)

set_target_properties(
        CNN
        PROPERTIES
        CUDA_PROPAGATE_HOST_FLAGS ON
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CNN cuda cudart cublas curand)

#[===[

target_link_libraries(CNN kernels)
add_library(kernels
        src/gpu.cu
        include/gpu.cuh
        include/operations/im2col_gpu.cuh
        include/operations/gemm_gpu.cuh
        src/opeartions/gemm_gpu.cu
        src/opeartions/im2col_gpu.cu)


target_link_libraries(kernels cuda cudart cublas curand)

set_target_properties(
        kernels
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


]===]

