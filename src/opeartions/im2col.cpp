#include "operations/im2col.h"
#include <vector>

template <typename T>
T get_im_value(const std::vector<T> &data, int row, int col, int channel, int width, int height)
{
    int img_index = (channel * height + row ) * width + col;
    if (row >= height || row < 0 || col >= width || col < 0)
        return 0;
    else
        return data[img_index];
}

template <typename T>
void im2col(const std::vector<T> &source, int width, int height, int channels,
            int kernel, int pad, int stride, std::vector<T> &col)
{
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
                col[dst_index] = get_im_value<T>(source, h_img, w_img, c_img, width, height);
            }
        }
    }
}
