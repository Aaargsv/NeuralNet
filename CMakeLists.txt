cmake_minimum_required(VERSION 3.17)
project(CNN)

set(CMAKE_CXX_STANDARD 14)

add_executable(CNN main.cpp layer.h network.cpp network.h neural.h activation_layer.h leaky_relu_layer.h convolution_layer.h convolution_layer.cpp max_pooling_layer.h max_pooling_layer.cpp batch_norm_layer.h batch_norm_layer.cpp utils.h utils.cpp bias_layer.h)