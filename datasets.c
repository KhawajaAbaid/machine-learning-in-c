#include "datasets.h"


void free_dataset(dataset *ds)
{
    free(ds->x);
    free(ds->y);
    free(ds);
}

/*
 * Rain in Australia dataset
 * Ref: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
 * I preprocessed it and saved it in binary format.
 */
dataset *load_weather()
{
    const size_t n_samples = 24800;
    const size_t n_features = 22;

    double *x = (double *) malloc(n_samples * n_features * sizeof(double));
    double *y = (double *) malloc(n_samples * sizeof(double));
    size_t ret;
    FILE *fp;
    fp = fopen("./datasets/weather_aus_x.bin", "rb");
    if (!fp)
    {
        printf("Error: Failed to load weather_aus_x.bin\n");
        exit(EXIT_FAILURE);
    }
    ret = fread(x, sizeof(double), n_samples * n_features, fp);
    fclose(fp); 
    if (ret != n_samples * n_features)
    {
        printf("Error: Failed to read weather_aus_x data.\n");
        exit(EXIT_FAILURE);
    }

    fp = fopen("./datasets/weather_aus_y.bin", "rb");
    if (!fp)
    {
        printf("Error: Failed to load weather_aus_y.bin\n");
        exit(EXIT_FAILURE);
    }
    ret = fread(y, sizeof(double), n_samples, fp);
    fclose(fp);
    if (ret != n_samples)
    {
        printf("Error: Failed to read file weather_aus_y.\n");
        exit(EXIT_FAILURE);
    }


    dataset *d = (dataset *) malloc(sizeof(dataset));
    d->x = x;
    d->y = y;
    d->n_samples = n_samples;
    d->dim = n_features;

    return d;
}


/*
 * MNIST dataset. Load the original mnsit dataset from binary format.
 * Then preprocess it to 1. Normalize images 2. One hot labels
 */

double *normalize_mnist_images(unsigned char *images, size_t image_size, 
        size_t n_samples, int for_gan)
{
    double *images_normalized = calloc(image_size * n_samples, sizeof(double));
    for (size_t i = 0; i < (image_size * n_samples); i++)
    {
        if (for_gan)
        {
            // -1 to 1 range
            images_normalized[i] = (images[i] - 127.5f) / 127.5f;
        }
        else
        {
            // 0 - 1 range
            images_normalized[i] = images[i] / 255.0f;
        }
    }
    return images_normalized;
}


double *one_hot(unsigned char *labels, size_t n_classes, size_t n_samples)
{
    double *one_hot_labels = calloc(n_samples * n_classes, sizeof(double));
    size_t label;
    for (size_t i = 0; i < n_samples; i++)
    {
        label = (size_t) labels[i];
        one_hot_labels[i * n_classes + label] =  1.0f;
    }

    return one_hot_labels;
}


dataset *load_mnist(int for_gan)
{
    const size_t n_images = 60000;
    const size_t image_size = 28 * 28;

    unsigned char *images = malloc(n_images * image_size);
    unsigned char *labels = malloc(n_images * sizeof(unsigned char));

    FILE *fp;
    size_t ret;
    
    fp = fopen("./datasets/mnist_x.bin", "rb");
    if (fp == NULL)
    {
        printf("Error: Failed to load mnist_x.bin file.\n");
        exit(EXIT_FAILURE);
    }
    ret = fread(images, sizeof(unsigned char), n_images * image_size, fp);
    if (ret != n_images * image_size)
    {
        printf("Error: Failed to read mnist_x data.\n");
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("./datasets/mnist_y.bin", "rb");
    if (fp == NULL)
    {
        printf("Error: Failed to load mnist_y.bin file.\n");
        exit(EXIT_FAILURE);
    }
    ret = fread(labels, sizeof(unsigned char), n_images, fp);
    if (ret != n_images)
    {
        printf("Error: Failed to read mnist_y file.\n");
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    // normalize input values
    double *images_normlaized = normalize_mnist_images(images, image_size,
            n_images, for_gan);
    free(images);

    double *one_hot_labels = one_hot(labels, 10, n_images);
    free(labels);

    dataset *ds = malloc(sizeof(dataset));
    ds->x = images_normlaized;
    if (!for_gan)
    {
        ds->y = one_hot_labels;
    }
    ds->n_samples = n_images;
    ds->dim = image_size;
    return ds;
}

