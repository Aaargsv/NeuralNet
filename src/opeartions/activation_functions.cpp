#include "operations/activation_functions.h"
#include <cmath>
float leaky_relu(float x)
{
    return x > 0 ? x : 0.1 * x;
}

float logistic(float x)
{
    return 1.0 / (1.0 + std::exp(-x));
}
