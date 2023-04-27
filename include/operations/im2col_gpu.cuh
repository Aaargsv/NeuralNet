#ifndef IM2COL_GPU_H
#define IM2COL_GPU_H

void im2col_gpu_kernel(const int n, const float* data_im,
                       const int height, const int width, const int ksize,
                       const int pad,
                       const int stride,
                       const int height_col, const int width_col,
                       float *data_col);

void im2col_gpu(float *im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float *data_col);

#endif //IM2COL_GPU_H
