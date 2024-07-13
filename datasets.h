#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>


typedef struct
{
    float *x;
    float *y;
    size_t n_samples;
    size_t dim;
} dataset;


dataset *load_weather();
