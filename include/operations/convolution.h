#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>


void convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                 int stride, int pad, std::vector<float> &weights , int number_filters,
                 std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst);

void winograd_convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                          int stride, int pad, std::vector<float> &weights , int number_filters,
                          std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst);

void kn2row_convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                        int stride, int pad, std::vector<float> &weights , int number_filters,
                        std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst);

void convolution_gpu(float *dev_src, int channels, int src_height, int src_width, int kernel,
                     int stride, int pad, float *dev_weights, int number_filters,
                     float *dev_utility_memory, int dst_height, int dst_width, float *dev_dst);

#endif //CONVOLUTION_H
