#ifndef MAX_POOL_H
#define MAX_POOL_H
#include <vector>

template <typename T>
void max_pool(const std::vector<T> &src, int channels, int src_height, int src_width,
              int kernel, int stride, int pad, std::vector<T> &dst);

#endif //MAX_POOL_H
