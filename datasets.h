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
 
float *normalize_mnist_images(unsigned char *images, size_t image_size, 
        size_t n_samples);

float *one_hot(unsigned char *labels, size_t n_classes, size_t n_samples);

dataset *load_mnist();
