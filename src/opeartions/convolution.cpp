#include "operations/convolution.h"
#include "operations/im2col_gpu.h"
#include "operations/gemm.h"
#include "operations/im2col_gpu.cuh"
#include "operations/gemm_gpu.cuh"
#include "gpu.cuh"
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

void convolution_gpu(float *dev_src, int channels, int src_height, int src_width, int kernel,
                 int stride, int pad, float *dev_weights, int number_filters,
                 float *dev_utility_memory, int dst_height, int dst_width, float *dev_dst)
{

    im2col_gpu(dev_src, channels, src_height, src_width, kernel, stride, pad, dev_utility_memory);

    int M = number_filters;
    int N = dst_height * dst_width;
    int K = kernel * kernel * channels;

    std::cout << "M * K = " << M * K << std::endl;

    gemm_gpu(0, 0, M, N, K, 1, dev_weights, K, dev_utility_memory, N, 1, dev_dst, N);

}