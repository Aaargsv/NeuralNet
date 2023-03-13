#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>
template <typename T>
void convolution(std::vector<T> src, int channels, int src_height, int src_width, int kernel,
                 int stride, int pad, std::vector<T> weights , int number_filters,
                 std::vector<T> utility_memory, int dst_height, int dst_width, std::vector<T> dst);

#endif //CONVOLUTION_H
