#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H
#include <vector>


void add_bias(std::vector<float> &tensor, const std::vector<float> &bias, int channels, int size);

void scale(std::vector<float> &tensor, const std::vector<float> &scales, int channels, int size);

void normalize(std::vector<float> &tensor, const std::vector<float> &rolling_mean,
               const std::vector<float> &rolling_variance, int channels, int size);

void concatenate(std::vector<float> &dst, const std::vector<float> &src);

void add_tensors(const std::vector<float> &a, const std::vector<float> &b,
                 int size, std::vector<float> &c);

void copy_vector(std::vector<float> &dst, const std::vector<float> &src, int start, int len);

#endif //TENSOR_MATH_H
