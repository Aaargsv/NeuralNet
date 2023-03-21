#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H
#include <vector>

template <typename T>
void add_bias(std::vector<T> &tensor, const std::vector<T> &bias, int channels, int size);

template <typename T>
void scale(std::vector<T> &tensor, const std::vector<T> &scales, int channels, int size);

void normalize(std::vector<float> &tensor, const std::vector<float> &rolling_mean,
               const std::vector<float> &rolling_variance, int channels, int size);

void concatenate(std::vector<float> &dst, const std::vector<float> &src);

void add_tensors(const std::vector<float> &a, const std::vector<float> &b,
                 int size, std::vector<float> &c);



#endif //TENSOR_MATH_H
