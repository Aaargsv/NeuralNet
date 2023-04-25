#include "operations/upsampling.h"


void upsample(std::vector<float> &src, int channels, int height, int width, int stride, std::vector<float> &dst)
{
    int dst_height = height * stride;
    int dst_width = width * stride;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < dst_height; h++) {
            for (int w = 0; w < dst_width; w++) {
                int src_index = c * height * width + h / stride * width  + w / stride;
                int dst_index = c * dst_height * dst_width + h * dst_width + w;
                dst[dst_index] = src[src_index];
            }
        }
    }
}