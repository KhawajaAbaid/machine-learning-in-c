#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>


typedef struct
{
    double *x;
    double *y;
    size_t n_samples;
    size_t dim;
} dataset;

void free_dataset(dataset *ds);

dataset *load_weather();
 
double *normalize_mnist_images(unsigned char *images, size_t image_size, 
        size_t n_samples, int for_gan);

double *one_hot(unsigned char *labels, size_t n_classes, size_t n_samples);

dataset *load_mnist(int for_gan);
