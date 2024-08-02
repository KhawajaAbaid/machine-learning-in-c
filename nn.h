#pragma once

#include <stdlib.h>
#include <stddef.h>
#include "random.h"
#include <math.h>


typedef struct
{
    double **w;
    double *b;
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
    double loss;
    double accuracy;
};

typedef struct
{
    gradient *grad;
    struct metrics *metrics;
} grad_and_metrics;


mlp *create_mlp(const size_t n_layers, const size_t *dims);
mlp *copy_mlp(mlp *source);

static inline void free_2d_matrix(double **m, const size_t n_rows)
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


double sigmoid(double x);
double sigmoid_prime(double x);
double crossentropy_loss(double *y, double *y_pred, const size_t dim);

void accumulate_grad_(mlp *accumulated_gradient, mlp* new_gradient);
void divide_grad_by_batch_size_(mlp *accumulated_gradient, size_t batch_size);
void update_weights_(mlp *model, mlp* gradient, double learning_rate);

void initialize_mlp_zeros_(mlp *model);
void initialize_mlp_normal_(const unsigned int seed, mlp *model);
void initialize_mlp_glorot_normal_(const unsigned int seed, mlp *model);
