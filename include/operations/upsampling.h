#ifndef UPSAMPLING_H
#define UPSAMPLING_H
#include <vector>

template <typename T>
void upsample(std::vector<T> &src, int channels, int height, int width, int stride, std::vector<T> &dst);


#endif //UPSAMPLING_H
