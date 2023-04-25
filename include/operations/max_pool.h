#ifndef MAX_POOL_H
#define MAX_POOL_H
#include <vector>


void max_pool(const std::vector<float> &src, int channels, int src_height, int src_width,
              int kernel, int stride, int pad, std::vector<float> &dst);

#endif //MAX_POOL_H
