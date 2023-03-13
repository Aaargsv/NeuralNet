#include <operations/tensor_math.h>
#include <vector>
#include <cmath>

template <typename T>
void add_bias(std::vector<T> &tensor, const std::vector<T> &bias, int channels, int size)
{
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < size; i++) {
            tensor[c * size + i] += bias[c];
        }
    }
}

template <typename T>
void scale(std::vector<T> &tensor, const std::vector<T> &scales, int channels, int size)
{
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < size; i++) {
            tensor[c * size + i] *= scales[c];
        }
    }
}

void normalize(std::vector<float> &tensor, const std::vector<float> &rolling_mean,
               const std::vector<float> &rolling_variance, int channels, int size)
{
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < size; i++) {
            int index = c * size + i;
            tensor[index] = (tensor[index] - rolling_mean[c]) /
                    (sqrtf(rolling_variance[c]) + 0.000001f);
        }
    }
}

