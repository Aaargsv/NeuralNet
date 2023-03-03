#ifndef IM2COL_H
#define IM2COL_H
#include <vector>

template <typename T>
T get_im_value(std::vector<T> *data, int row, int col, int channel, int width, int height);

template <typename T>
void im2col(std::vector<T> *source, int width, int height, int channels,
            int kernel, int pad, int stride, std::vector<T> *col);

#endif //IM2COL_H
