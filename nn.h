#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <stdarg.h>
#include "random.h"
#include "ops.h"

/*
 * ----------- STRUCTS -----------
 */
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

typedef struct
{
    double **z_all;
    double **a_all;

} outputs_with_logits;

/*
 * ----------- FREE MEMORY -----------
 */
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


/*
 * ---------------- MLP RELATED FUNCS ----------------
 */

mlp *create_mlp(const size_t n_layers, const size_t *dims);
mlp *copy_mlp(mlp *source);

void accumulate_grad_(mlp *accumulated_gradient, mlp* new_gradient);
void divide_grad_by_batch_size_(mlp *accumulated_gradient, size_t batch_size);
void update_weights_(mlp *model, mlp* gradient, double learning_rate);

void initialize_mlp_zeros_(mlp *model);
void initialize_mlp_normal_(const unsigned int seed, mlp *model);
void initialize_mlp_glorot_normal_(const unsigned int seed, mlp *model);


/*
 * ---------------- ACTIVATION FUNCTIONS ----------------
 */

// This is perhaps the better way of defining such activation functions
// so they can operate both on arrays and scalars.
// May soon do this for all other such functions.
static inline double *sigmoid_vectorized(double *x, const size_t dim);
static inline double sigmoid_scalar(double x);

#define sigmoid(x, ...)  _Generic((x), \
                                  double*: sigmoid_vectorized, \
                                  double: sigmoid_scalar \
                                  )(x, __VA_ARGS__)

static inline double *sigmoid_vectorized(double *x, const size_t dim)
{
    double *result = (double *)calloc(dim, sizeof(double));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = 1.0 / (1.0 + exp(-x[i]));
    }
    return result;
}

static inline double sigmoid_scalar(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline double *sigmoid_prime(double *x, const size_t dim)
{
    double *result = (double *)malloc(dim * sizeof(double));

    double *sig = sigmoid_vectorized(x, dim);
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = sig[i] * (1.0 - sig[i]); 
    }
    return result;
}

static inline double *leaky_relu(double *x,
                                 double negative_slope, 
                                 const size_t dim)
{
    double *result = (double *)malloc(dim * sizeof(double));
    double r;
    for (size_t i = 0; i < dim; i++)
    {
        r = (x[i] > 0.0) ? x[i] : 0.0;
        r += negative_slope * ((x[i] < 0) ? x[i] : 0.0);
        result[i] = r;
    }
    return result;
}


static inline double *leaky_relu_prime(double *x,
                                       double negative_slope, 
                                       const size_t dim)
{
    double *result = (double *)malloc(dim * sizeof(double));
    double r;
    for (size_t i = 0; i < dim; i++)
    {
        r = (x[i] > 0.0) ? 1.0f : 0.0;
        r += negative_slope * ((x[i] < 0) ? 1.0f : 0.0);
        result[i] = r;
    }
    return result;
}

// we name it vectorized since math.h contains a definition for tanh
static inline double *tanh_vectorized(double *x, const size_t dim)
{
    double *result = (double *)malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = tanh(x[i]);
    }
    return result;
}

static inline double *tanh_prime(double *x, const size_t dim)
{
    double *result = (double *)malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = 1.0f - square(tanh(x[i]));
    }
    return result;
}

/*
 * ---------------- LOSS FUNCTIONS ----------------
 */

static inline double crossentropy_loss(double *y,
                                       double *y_pred,
                                       const size_t dim)
{
    double loss = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        loss += y[i] * log(y_pred[i]) + (1.0 - y[i]) * log(1.0 - y_pred[i]); 
    }
    return -loss;
}

static inline double binary_crossentropy_loss(double y, double y_pred)
{
    return -((y * log(y_pred)) + ((1.0f - y) * log(1.0f - y_pred)));
}


