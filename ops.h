#pragma once

#include <stdlib.h>


static inline double square(double x)
{
    return x * x;
}

static inline size_t argmax(double *x, const size_t dim)
{
    size_t max_idx = 0;
    double max = 0.0f;

    for (size_t i = 0; i < dim; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}
