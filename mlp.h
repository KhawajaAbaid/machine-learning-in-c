#pragma once

#include <stdlib.h>
#include <stddef.h>
#include "random.h"


typedef struct
{
    float **w;
    float *b;
    size_t in_dim;
    size_t out_dim;
} layer;


typedef struct
{
    layer **layers;
    size_t n_layers;
} mlp;

// gradient has same structure as the model
typedef mlp gradient;

struct metrics
{
    float loss;
    float accuracy;
};

typedef struct
{
    gradient *grad;
    struct metrics *metrics;
} grad_and_metrics;


// In our model, there's no standalone "input" layer.
mlp *create_mlp(const size_t n_layers, const size_t *dims)
{
    // dims array contains 1 more element than the number of layers
    mlp *model = (mlp *) malloc(sizeof(mlp));
    model->layers = (layer **) malloc(n_layers * sizeof(layer *));
    model->n_layers = n_layers;

    for (size_t i = 0; i < n_layers; i++)
    {
        model->layers[i] = (layer *) malloc(sizeof(layer));
        model->layers[i]->in_dim = dims[i];
        model->layers[i]->out_dim = dims[i + 1];
    }

    return model;
}


// Creates copy of a given mlp of the same structure but without weights
mlp *copy_mlp(mlp *source)
{
    mlp *copied = (mlp *) malloc(sizeof(mlp));
    copied->layers = (layer **) malloc(source->n_layers * sizeof(layer *));
    copied->n_layers = source->n_layers;
    // set copied layers dims
    for (size_t i = 0; i < source->n_layers; i++)
    {
        copied->layers[i] = (layer *) malloc(sizeof(layer));
        copied->layers[i]->out_dim = source->layers[i]->out_dim;
        copied->layers[i]->in_dim = source->layers[i]->in_dim;
    }
    return copied;
}


void initialize_weights(const unsigned int seed, mlp *model)
{
    srand(seed);

    float **w;
    float *b;
    layer *l;

    for (size_t i = 0; i < model->n_layers; i++)
    {   
        l = model->layers[i];
        w = (float **) malloc(l->out_dim * sizeof(float *));
        b = (float *) calloc(l->out_dim, sizeof(float));
        for (size_t j = 0; j < l->out_dim; j++)
        {
            w[j] = (float *) malloc(l->in_dim * sizeof(float));

            for (size_t k = 0; k < l->in_dim; k++)
            {
                w[j][k] = random_normal();
            }
            b[j] = random_normal();
        }
        l->w = w;
        l->b = b;
    }
}


static inline void free_2d_matrix(float **m, const size_t n_rows)
{
    for (size_t i = 0; i < n_rows; i++)
    {
        free(m[i]);
    }
    free(m);
}


static inline void free_layer(layer *l)
{
    free_2d_matrix(l->w, l->out_dim);
    free(l->b);
    free(l);
}


static inline void free_mlp(mlp *m)
{
    for (size_t i = 0; i < m->n_layers; i++)
    {
        free_layer(m->layers[i]);
    }
    free(m->layers);
    free(m);
}


static inline void free_gradient(gradient *grad)
{
    free_mlp(grad);
}


static inline void free_grad_and_metrics(grad_and_metrics *gam)
{
    free_mlp(gam->grad);
    free(gam->metrics);
    free(gam);
}


void accumulate_grad(mlp *accumulated_gradient, mlp* new_gradient)
{
    for (size_t i = 0; i < accumulated_gradient->n_layers; i++)
    {
        for (size_t j = 0; j < accumulated_gradient->layers[i]->out_dim; j++)
        {
            for (size_t k = 0; k < accumulated_gradient->layers[i]->in_dim; k++)
            {
                accumulated_gradient->layers[i]->w[j][k] += new_gradient->layers[i]->w[j][k];
            }
            accumulated_gradient->layers[i]->b[j] += new_gradient->layers[i]->b[j];
        }
    }
}


void update_weights(mlp *model, mlp* gradient, float learning_rate)
{
    for (size_t i = 0; i < model->n_layers; i++)
    {
        for (size_t j = 0; j < model->layers[i]->out_dim; j++)
        {
            for (size_t k = 0; k < model->layers[i]->in_dim; k++)
            {
                model->layers[i]->w[j][k] -= learning_rate * gradient->layers[i]->w[j][k];
            }
            model->layers[i]->b[j] -= learning_rate * gradient->layers[i]->b[j];
        }
    }
}
