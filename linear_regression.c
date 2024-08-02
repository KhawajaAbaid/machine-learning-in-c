/*
 * Simplest linear regression.
 */

#include "linalg.h"
#include <stdio.h>
#include <stdlib.h>


typedef struct 
{
    float w;
    float b;
} regressor;


typedef struct
{
    float *x;
    float *y;
    size_t n_samples;
} dataset;


void initialize_weights_(unsigned int seed, regressor *reg)
{
    srand(seed);
    reg->w = (float) (rand() % 10);
    reg->b = (float) (rand() % 10);
}


float square(float x)
{
    return x * x;
}


float mse(float *y, float *y_pred, const size_t dim)
{
    float loss = 0.0f;
    for (size_t i = 0; i < dim; i++)
    {
        loss += square(y[i] - y_pred[i]) / 2.0f;
    }
    loss /= dim;
    return loss;
}


float *forward(regressor *reg, dataset *data)
{
    float *y_pred = calloc(data->n_samples, sizeof(float));

    for (size_t i = 0; i < data->n_samples; i++)
    {
        y_pred[i] = reg->w * data->x[i] + reg->b;
    }
    return y_pred;
}

// Gradient of weights and bias w.r.t. loss
regressor *grad(float *x, float *y, float *y_pred, const size_t n_samples)
{
    float *dloss_dypred = vector_subtraction(y_pred, y, n_samples);
    float *dypred_dw = x;
    float *dloss_dw = vector_elementwise_product(dloss_dypred, dypred_dw,
            n_samples);
    // The dloss_dw vector contains gradient for each input instance.
    // We need to sum it all.
    float dloss_dw_total = vector_sum(dloss_dw, n_samples);
    
    float *dloss_db = dloss_dypred;
    float dloss_db_total = vector_sum(dloss_db, n_samples);

    free(dloss_dw);
    free(dloss_db);

    regressor *gradient = calloc(1, sizeof(regressor));
    gradient->w = dloss_dw_total;
    gradient->b = dloss_db_total;

    return gradient;
}

static inline void update_weights_(regressor *reg, regressor *gradient,
        float learning_rate)
{
    reg->w -= learning_rate * gradient->w;
    reg->b -= learning_rate * gradient->b;
}


float train_step(regressor *reg, dataset *data, float learning_rate)
{
    // Run forward pass
    float *y_pred = forward(reg, data);
    
    // Compute loss
    float loss = mse(data->y, y_pred, data->n_samples);

    // Compute gradient w.r.t. loss
    regressor *gradient = grad(data->x, data->y, y_pred, data->n_samples);
    
    // Update weights
    update_weights_(reg, gradient, learning_rate);

    return loss;
}


void fit(regressor *reg, dataset *data, float learning_rate, const size_t n_epochs)
{
    float loss;

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
    
    float *x = calloc(n_samples, sizeof(float));
    float *y = calloc(n_samples, sizeof(float));

    for (size_t i = 0; i < n_samples; i++)
    {
        x[i] = (float) rand() / (float) RAND_MAX;
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
    float learning_rate = 0.001f;

    fit(reg, data, learning_rate, 100);
}
