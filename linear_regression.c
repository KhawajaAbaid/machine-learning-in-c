/*
 * Simplest linear regression.
 */

#include "linalg.h"
#include <stdio.h>
#include <stdlib.h>


typedef struct 
{
    double w;
    double b;
} regressor;


typedef struct
{
    double *x;
    double *y;
    size_t n_samples;
} dataset;


void initialize_weights_(unsigned int seed, regressor *reg)
{
    srand(seed);
    reg->w = (double) (rand() % 10);
    reg->b = (double) (rand() % 10);
}


double square(double x)
{
    return x * x;
}


double mse(double *y, double *y_pred, const size_t dim)
{
    double loss = 0.0f;
    for (size_t i = 0; i < dim; i++)
    {
        loss += square(y[i] - y_pred[i]) / 2.0f;
    }
    loss /= dim;
    return loss;
}


double *forward(regressor *reg, dataset *data)
{
    double *y_pred = calloc(data->n_samples, sizeof(double));

    for (size_t i = 0; i < data->n_samples; i++)
    {
        y_pred[i] = reg->w * data->x[i] + reg->b;
    }
    return y_pred;
}

// Gradient of weights and bias w.r.t. loss
regressor *grad(double *x, double *y, double *y_pred, const size_t n_samples)
{
    double *dloss_dypred = vector_subtraction(y_pred, y, n_samples);
    double *dypred_dw = x;
    double *dloss_dw = vector_elementwise_product(dloss_dypred, dypred_dw,
            n_samples);
    // The dloss_dw vector contains gradient for each input instance.
    // We need to sum it all.
    double dloss_dw_total = vector_sum(dloss_dw, n_samples);
    
    double *dloss_db = dloss_dypred;
    double dloss_db_total = vector_sum(dloss_db, n_samples);

    free(dloss_dw);
    free(dloss_db);

    regressor *gradient = calloc(1, sizeof(regressor));
    gradient->w = dloss_dw_total;
    gradient->b = dloss_db_total;

    return gradient;
}

static inline void update_weights_(regressor *reg, regressor *gradient,
        double learning_rate)
{
    reg->w -= learning_rate * gradient->w;
    reg->b -= learning_rate * gradient->b;
}


double train_step(regressor *reg, dataset *data, double learning_rate)
{
    // Run forward pass
    double *y_pred = forward(reg, data);
    
    // Compute loss
    double loss = mse(data->y, y_pred, data->n_samples);

    // Compute gradient w.r.t. loss
    regressor *gradient = grad(data->x, data->y, y_pred, data->n_samples);
    
    // Update weights
    update_weights_(reg, gradient, learning_rate);

    return loss;
}


void fit(regressor *reg, dataset *data, double learning_rate, const size_t n_epochs)
{
    double loss;

    for (size_t i = 0; i < n_epochs; i++)
    {
        loss = train_step(reg, data, learning_rate);

        printf("Epoch: %ld/%ld | Loss: %.5f | w: %.4f | b: %.4f \n",
                i + 1, n_epochs, loss, reg->w, reg->b);
    }
}


dataset *generate_input_data(unsigned int seed, const size_t n_samples)
{
    srand(seed);
    
    double *x = calloc(n_samples, sizeof(double));
    double *y = calloc(n_samples, sizeof(double));

    for (size_t i = 0; i < n_samples; i++)
    {
        x[i] = (double) rand() / (double) RAND_MAX;
        y[i] = x[i] * 5.0f + 10.0f;
    }

    dataset *data = malloc(sizeof(dataset));
    data->x = x;
    data->y = y;
    data->n_samples = n_samples;

    return data;
}


void main()
{
    dataset *data = generate_input_data(1337, 256);
    regressor *reg = malloc(sizeof(regressor));
    initialize_weights_(999, reg);
    double learning_rate = 0.001f;

    fit(reg, data, learning_rate, 100);
}
