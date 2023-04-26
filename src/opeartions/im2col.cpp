#include "operations/im2col.h"
#include <vector>
#include <iostream>


float get_im_value(const std::vector<float> &data, int row, int col, int channel, int width, int height)
{
    int img_index = (channel * height + row ) * width + col;
    if (row >= height || row < 0 || col >= width || col < 0)
        return 0;
    else
        return data[img_index];
}


void im2col(const std::vector<float> &source, int width, int height, int channels,
            int kernel, int pad, int stride, std::vector<float> &col)
{
    std::cout << "col_matrix capacity = " << col.capacity() << std::endl;
    int width_col = (width + 2 * pad - kernel) / stride + 1;
    int height_col =  (height + 2 * pad - kernel) / stride + 1;
    int channels_col = channels * kernel * kernel;
    for (int c = 0; c < channels_col; c++) {
        int width_offset = c % kernel;
        int height_offset = c / kernel % kernel;
        int c_img = c / kernel / kernel;
        for (int h = 0; h < height_col; h++) {
            int h_img = h * stride + height_offset - pad;
            for (int w = 0; w < width_col; w++) {
                int w_img = w * stride + width_offset - pad;
                int dst_index =  (c * height_col + h) * width_col + w;
                col[dst_index] = get_im_value(source, h_img, w_img, c_img, width, height);
            }
        }
    }
}
