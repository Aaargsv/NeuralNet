#include "operations/convolution.h"
#include "operations/im2col.h"
#include "operations/gemm.h"
#include "operations/im2col_gpu.cuh"
#include "operations/gemm_gpu.cuh"
#include "gpu.cuh"
#include <vector>
#include <assert.h>
#include <iostream>


template <typename T>
T get_img_value(T *data, int row, int col, int channel, int width, int height)
{
    int img_index = (channel * height + row ) * width + col;
    if (row >= height || row < 0 || col >= width || col < 0)
        return 0;
    else
        return data[img_index];
}


template <typename T, typename T1>
void set_output1(T *src, T1 *dst, int size, int stride)
{
    T1 tmp[8];
    tmp[0] = src[0 * size] + src[1 * size] + src[2 * size];
    tmp[1] = src[1 * size] - src[2 * size] - src[3 * size];
    tmp[2] = src[4 * size] + src[5 * size] + src[6 * size];
    tmp[3] = src[5 * size] - src[6 * size] - src[7 * size];
    tmp[4] = src[8 * size] + src[9 * size] + src[10 * size];
    tmp[5] = src[9 * size] - src[10 * size] - src[11 * size];
    tmp[6] = src[12 * size] + src[13 * size] + src[14 * size];
    tmp[7] = src[13 * size] - src[14 * size] - src[15 * size];
    dst[0] = tmp[0] + tmp[2] + tmp[4];
    dst[1] = tmp[1] + tmp[3] + tmp[5];
    dst[stride + 0] = tmp[2] - tmp[4] - tmp[6];
    dst[stride + 1] = tmp[3] - tmp[5] - tmp[7];
}

template <typename T, typename T1>
void set_output1p(T *src, T1 *dst, int size,
                  int dst_width, int row_limit, int col_limit)
{
    T1 tmp[4];
    set_output1(src, tmp, size, 2);
    for (int row = 0; row < row_limit; row++) {
        for (int col = 0; col < col_limit; col++) {
            dst[row * dst_width + col] = tmp[row * 2 + col];
        }
    }
}


template <typename T>
void set_output(T * src, int size, T * dst,
                int dst_channels, int dst_height, int dst_width,
                int dst_height_floor, int dst_width_floor)
{
    int row_limit = dst_height - dst_height_floor;
    int col_limit = dst_width - dst_width_floor;
    for (int c = 0; c < dst_channels; c++) {
        int h, w;
        for (h = 0; h < dst_height_floor; h += 2) {
            for (w = 0; w < dst_width_floor; w += 2) {
                set_output1(src++, dst + h * dst_width + w, size, dst_width);
            }
            if (w < dst_width) {
                set_output1p(src++, dst + h * dst_width + w, size,
                             dst_width, 2, col_limit);
            }
        }
        if (h < dst_height) {
            for (w = 0; w < dst_width_floor; w += 2) {
                set_output1p(src++, dst + h * dst_width + w, size,
                             dst_width, row_limit, 2);

            }
            if (w < dst_width) {
                set_output1p(src++, dst + h * dst_width + w, size,
                             dst_width, row_limit, col_limit);
            }
        }
        dst += dst_height * dst_width;
    }
}

template <typename T>
void set_filter(T *src, int size, T *dst)
{
    for (int i = 0; i < size; i++, src += 9, dst += 1) {
        dst[0 * size] = src[0];
        dst[1 * size] = (src[0] + src[2] + src[1]) / 2;
        dst[2 * size] = (src[0] + src[2] - src[1]) / 2;
        dst[3 * size] = src[2];
        dst[4 * size] = (src[0] + src[6] + src[3]) / 2;
        dst[5 * size] = ((src[0] + src[6] + src[3]) +
                         (src[2] + src[8] + src[5]) + (src[1] + src[7] + src[4])) / 4;
        dst[6 * size] = ((src[0] + src[6] + src[3]) +
                         (src[2] + src[8] + src[5]) - (src[1] + src[7] + src[4])) / 4;
        dst[7 * size] = (src[2] + src[8] + src[5]) / 2;
        dst[8 * size] = (src[0] + src[6] - src[3]) / 2;
        dst[9 * size] = ((src[0] + src[6] - src[3]) +
                         (src[2] + src[8] - src[5]) + (src[1] + src[7] - src[4])) / 4;
        dst[10 * size] = ((src[0] + src[6] - src[3]) +
                          (src[2] + src[8] - src[5]) - (src[1] + src[7] - src[4])) / 4;
        dst[11 * size] = (src[2] + src[8] - src[5]) / 2;
        dst[12 * size] = src[6];
        dst[13 * size] = (src[6] + src[8] + src[7]) / 2;
        dst[14 * size] = (src[6] + src[8] - src[7]) / 2;
        dst[15 * size] = src[8];
    }
}



template <typename T>
void set_input(T *src, int channels, int src_height, int src_width,
               int dst_height, int dst_width, int pad, T *dst, int size)
{
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < dst_height; h += 2) {
            for (int w = 0; w < dst_width; w += 2) {
                T tmp[16];
                for (int win_y = 0; win_y < 4; win_y++) {
                    for (int win_x = 0; win_x < 4; win_x++) {
                        int row = h + win_y - pad;
                        int col = w + win_x - pad;
                        tmp[win_y * 4 + win_x] = get_img_value(src, row, col, c,
                                                               src_width, src_height);
                    }
                }
                dst[0 * size] = (tmp[0] - tmp[8]) - (tmp[2] - tmp[10]);
                dst[1 * size] = (tmp[1] - tmp[9]) + (tmp[2] - tmp[10]);
                dst[2 * size] = (tmp[2] - tmp[10]) - (tmp[1] - tmp[9]);
                dst[3 * size] = (tmp[1] - tmp[9]) - (tmp[3] - tmp[11]);
                dst[4 * size] = (tmp[4] + tmp[8]) - (tmp[6] + tmp[10]);
                dst[5 * size] = (tmp[5] + tmp[9]) + (tmp[6] + tmp[10]);
                dst[6 * size] = (tmp[6] + tmp[10]) - (tmp[5] + tmp[9]);
                dst[7 * size] = (tmp[5] + tmp[9]) - (tmp[7] + tmp[11]);
                dst[8 * size] = (tmp[8] - tmp[4]) - (tmp[10] - tmp[6]);
                dst[9 * size] = (tmp[9] - tmp[5]) + (tmp[10] - tmp[6]);
                dst[10 * size] = (tmp[10] - tmp[6]) - (tmp[9] - tmp[5]);
                dst[11 * size] = (tmp[9] - tmp[5]) - (tmp[11] - tmp[7]);
                dst[12 * size] = (tmp[4] - tmp[12]) - (tmp[6] - tmp[14]);
                dst[13 * size] = (tmp[5] - tmp[13]) + (tmp[6] - tmp[14]);
                dst[14 * size] = (tmp[6] - tmp[14]) - (tmp[5] - tmp[13]);
                dst[15 * size] = (tmp[5] - tmp[13]) - (tmp[7] - tmp[15]);
                dst++;
            }
        }
    }
}


void winograd_convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                          int stride, int pad, std::vector<float> &weights , int number_filters,
                          std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst)
{
    const int block = 2;
    int count = (block + kernel - 1) * (block + kernel - 1);
    int dst_height_floor =  dst_height / 2 * 2;
    int dst_width_floor =  dst_width / 2 * 2;
    int tile_h = (dst_height + 1) / block;
    int tile_w = (dst_width + 1) / block;
    int size_w = channels * number_filters;
    int size_s = channels * tile_h * tile_w;
    int size_d;

    size_d = number_filters * tile_h * tile_w;

    int M = number_filters;
    int N = tile_h * tile_w;
    int K = channels;

    float *buf_w = utility_memory.data();
    float *buf_s = buf_w + size_w * count;
    float *buf_d = buf_s + size_s * count;

    set_filter(weights.data(), size_w, buf_w);

    set_input(src.data(), channels, src_height, src_width,
              dst_height, dst_width, pad,
              buf_s, size_s);


    for (int i = 0; i < count; i++) {
        gemm(M, N, K, buf_w + i * size_w, K, buf_s + i * size_s, N, buf_d + i * size_d, N);
    }

    set_output(buf_d, size_d, dst.data(),
               number_filters, dst_height, dst_width,
               dst_height_floor, dst_width_floor);

}


void axpy(int size, float *x, float *y)
{
    for (int i = 0; i < size; i++)
        y[i] += x[i];
}

void matrix_shift_add(float *base_mat,
                    int base_no_rows, int base_no_cols,
                    float *overlap_mat,
                    int ov_no_rows, int ov_no_cols,
                    int row_shift, int col_shift) {
    if (row_shift == 0 && col_shift == 0 && (base_no_rows == ov_no_rows) &&
        (base_no_cols == ov_no_cols)) {

        axpy(base_no_rows * base_no_cols, overlap_mat, base_mat);
        return;
    }
    int rows_to_add, cols_to_add;
    int base_row_start, base_col_start;
    int ov_row_start, ov_col_start;

    if (ov_no_rows > base_no_rows) {
        rows_to_add = base_no_rows;
        cols_to_add = base_no_cols;
        base_row_start = 0;
        base_col_start = 0;
        ov_row_start = row_shift < 0? -row_shift : 0;
        ov_col_start = col_shift < 0? -col_shift : 0;

    } else {
        rows_to_add = ov_no_rows - abs(row_shift);
        cols_to_add = ov_no_cols - abs(col_shift);

        ov_col_start = col_shift > 0? col_shift : 0;
        ov_row_start = row_shift > 0? row_shift : 0;
        base_row_start = row_shift < 0? -row_shift : 0;
        base_col_start = col_shift < 0? -col_shift : 0;
    }

    for (int r = 0; r < rows_to_add; ++r) {
        int base_mat_offset = (r + base_row_start) * base_no_cols + base_col_start;
        int overlap_mat_offset = (r + ov_row_start) * ov_no_cols + ov_col_start;
        axpy(cols_to_add,  overlap_mat + overlap_mat_offset,
             base_mat + base_mat_offset);
    }
}


void weights_NCHW_2_HWNC(float *src_weights, int kernel, int num_filters, int channels, float *dst_weights)
{
    for (int n = 0; n < num_filters; n++) {
        for (int c = 0; c < channels; c++) {
            for (int k = 0; k < kernel * kernel; k++) {
                int dst_index = k * channels * num_filters + n * channels + c;
                int src_index = n * channels * kernel * kernel + c * kernel * kernel + k;
                dst_weights[dst_index] = src_weights[src_index];
            }
        }
    }
}

void kn2row_convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                        int stride, int pad, std::vector<float> &weights , int number_filters,
                        std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst)
{

    assert((pad == 0) || (pad == kernel / 2));
    assert(stride == 1);


    float *kknc_filters = utility_memory.data();
    float *gemm_output = kknc_filters + kernel * kernel * number_filters * channels;

    weights_NCHW_2_HWNC(weights.data(), kernel, number_filters, channels, kknc_filters);

    // Just for convenience
    int H = src_height;
    int W = src_width;
    float alpha = 1.0;
    float beta = 0.0;

    for (int kr = 0; kr < kernel; kr++) {
        int row_shift = kr - kernel / 2;
        for (int kc = 0; kc < kernel; kc++) {
            int group_no = kr * kernel + kc;
            int col_shift = kc - kernel / 2;
            // Matrix dimensions - A -> mxk B -> kxn  C --> mxn
            int m = number_filters;
            int k = channels;
            int n = src_height * src_width;
            // This is just 1x1 convolution

            gemm(m, n, k, kknc_filters + group_no * m * k, k, src.data(), n, gemm_output, n);

            for (int omap = 0; omap < number_filters; omap++) {
                matrix_shift_add( dst.data() + omap * dst_height * dst_width,
                               dst_height, dst_width,
                               gemm_output + omap * H * W,
                               H, W, row_shift, col_shift);
            }
        }
    }
}

void convolution(std::vector<float> &src, int channels, int src_height, int src_width, int kernel,
                 int stride, int pad, std::vector<float> &weights , int number_filters,
                 std::vector<float> &utility_memory, int dst_height, int dst_width, std::vector<float> &dst)
{
    std::vector<float> &col_matrix =  utility_memory;
    im2col(src, src_width, src_height, channels, kernel, pad, stride, col_matrix);
    int M = number_filters;
    int N = dst_height * dst_width;
    int K = kernel * kernel * channels;

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