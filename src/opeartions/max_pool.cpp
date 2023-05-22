#include "operations/max_pool.h"
#include <vector>
#include <algorithm>
#include <limits>


void max_pool(const std::vector<float> &src, int channels, int src_height, int src_width,
              int kernel, int stride, int pad, std::vector<float> &dst)
{
    /**
     * @brief
     * idk ...+ 2 * pad... or ...+ pad... in dst_height
     * and in dst_width
     * in darknet ...+ pad...
     */

    int dst_height = (src_height - kernel + pad) / stride + 1;
    int dst_width = (src_width - kernel + pad) / stride + 1;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < dst_height; h++) {
            int top = h * stride - pad;
            int bottom = std::min(top + kernel, src_height);
            top = std::max(top, 0);
            for (int w = 0; w < dst_width; w++) {
                int left = w * stride - pad;
                int right = std::min(left + kernel, src_width);
                left = std::max(left, 0);
                float max_value = std::numeric_limits<float>::lowest();
                for(int k_h = top; k_h < bottom; k_h++) {
                    for (int k_w = left; k_w < right; k_w++) {
                        int src_index = (c * src_height + k_h)  * src_width + k_w;
                        max_value = std::max(max_value, src[src_index]);
                    }
                }
                int dst_index = (c * dst_height + h) * dst_width + w;
                dst[dst_index] = max_value;
            }
        }
    }
}

void max_pool2(const std::vector<float> &src, int channels, int src_height, int src_width,
               int kernel, int stride, int pad, std::vector<float> &dst)
{
    int b,i,j,k,m,n;
    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int h = (src_height - kernel + pad) / stride + 1;
    int w = (src_width - kernel + pad) / stride + 1;
    int c = channels;
    b = 0;
    for(k = 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                int out_index = j + w*(i + h*(k + c*b));
                float max = -FLT_MAX;
                int max_i = -1;
                for(n = 0; n < kernel; ++n){
                    for(m = 0; m < kernel; ++m){
                        int cur_h = h_offset + i*stride + n;
                        int cur_w = w_offset + j*stride + m;
                        int index = cur_w + src_width*(cur_h + src_height*(k + b*channels));
                        int valid = (cur_h >= 0 && cur_h < src_height &&
                                     cur_w >= 0 && cur_w < src_width);
                        float val = (valid != 0) ? src[index] : -FLT_MAX;
                        max_i = (val > max) ? index : max_i;
                        max   = (val > max) ? val   : max;
                    }
                }
                dst[out_index] = max;
            }
        }
    }
}

