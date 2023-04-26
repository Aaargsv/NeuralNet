#include "operations/convolution.h"
#include "operations/im2col.h"
#include "operations/gemm.h"
#include <vector>
#include <assert.h>
#include <iostream>


void convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                 int stride, int pad, std::vector<float> &weights , int number_filters,
                 std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst)
{
    std::vector<float> &col_matrix =  utility_memory;
    im2col(src, src_width, src_height, channels, kernel, pad, stride, col_matrix);
    int M = number_filters;
    int N = dst_height * dst_width;
    int K = kernel * kernel * channels;

    std::cout << "M * K = " << M * K << std::endl;
    std::cout << "weights.capacity() = " << weights.capacity() << std::endl;
    assert(M * K == weights.capacity());

    gemm(M, N, K, weights, K, col_matrix, N, dst, N);
}