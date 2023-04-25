#ifndef UPSAMPLING_H
#define UPSAMPLING_H
#include <vector>


void upsample(std::vector<float> &src, int channels, int height, int width, int stride, std::vector<float> &dst);


#endif //UPSAMPLING_H
