#include "operations/convolution.h"
#include "operations/im2col.h"
#include "operations/gemm.h"
#include <vector>

template <typename T>
void convolution(std::vector<T> src, int channels, int src_height, int src_width, int kernel,
                 int stride, int pad, std::vector<T> weights , int number_filters,
                 std::vector<T> utility_memory, int dst_height, int dst_width, std::vector<T> dst)
{
    std::vector<T> &col_matrix =  utility_memory;
    im2col(src, src_width, src_height, channels, kernel, pad, stride, col_matrix);
    int M = number_filters;
    int N = dst_height * dst_width;
    int K = kernel * kernel * channels;
    gemm(M, N, K, weights, K, col_matrix, N, dst, N);
}