#include "datasets.h"


/*
 * Rain in Australia dataset
 * Ref: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
 * I preprocessed it and saved it in binary format.
 */
dataset *load_weather()
{
    const size_t n_samples = 24800;
    const size_t n_features = 22;

    float *x = (float *) malloc(n_samples * n_features * sizeof(float));
    float *y = (float *) malloc(n_samples * sizeof(float));
    size_t ret;
    FILE *fp;
    fp = fopen("./datasets/weather_aus_x.bin", "rb");
    if (!fp)
    {
        printf("Error: Failed to load weather_aus_x.bin\n");
        exit(EXIT_FAILURE);
    }
    ret = fread(x, sizeof(float), n_samples * n_features, fp);
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
    ret = fread(y, sizeof(float), n_samples, fp);
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
