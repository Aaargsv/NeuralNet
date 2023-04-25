#ifndef IM2COL_H
#define IM2COL_H
#include <vector>


float get_im_value(std::vector<float> *data, int row, int col, int channel, int width, int height);


void im2col(const std::vector<float> &source, int width, int height, int channels,
            int kernel, int pad, int stride, std::vector<float> &col);

#endif //IM2COL_H
