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
