#include "nn.h"
#include "random.h"


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


double sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_prime(double x)
{
    double sig = sigmoid(x);
    return sig * (1.0 - sig); 
}


double crossentropy_loss(double *y, double *y_pred, const size_t dim)
{
    double loss = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        loss += y[i] * log(y_pred[i]) + (1.0 - y[i]) * log(1.0 - y_pred[i]); 
    }
    return -loss;
}


void accumulate_grad_(mlp *accumulated_gradient, mlp* new_gradient)
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


// Divide sum of individual instance gradients by batch size to get avg
void divide_grad_by_batch_size_(mlp *accumulated_gradient, size_t batch_size)
{
    for (size_t i = 0; i < accumulated_gradient->n_layers; i++)
    {
        for (size_t j = 0; j < accumulated_gradient->layers[i]->out_dim; j++)
        {
            for (size_t k = 0; k < accumulated_gradient->layers[i]->in_dim; k++)
            {
                accumulated_gradient->layers[i]->w[j][k] /= (double) batch_size;
            }
            accumulated_gradient->layers[i]->b[j] /= (double) batch_size;
        }
    }
}


void update_weights_(mlp *model, mlp* gradient, double learning_rate)
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


void initialize_mlp_zeros_(mlp *model)
{
    // Init accumulated gradient values to zero
    for (size_t i = 0; i < model->n_layers; i++)
    {
        model->layers[i]->w = (double **) malloc(
                model->layers[i]->out_dim * sizeof(double *));
        for (size_t j = 0; j < model->layers[i]->out_dim; j++)
        {
            model->layers[i]->w[j] = (double *) calloc(
                    model->layers[i]->in_dim,
                    sizeof(double));
        }
        model->layers[i]->b = (double *) calloc(
                model->layers[i]->out_dim,
                sizeof(double));
    }
}



void initialize_mlp_normal_(const unsigned int seed, mlp *model)
{
    srand(seed);

    double **w;
    double *b;
    layer *l;

    for (size_t i = 0; i < model->n_layers; i++)
    {   
        l = model->layers[i];
        w = (double **) malloc(l->out_dim * sizeof(double *));
        b = (double *) calloc(l->out_dim, sizeof(double));
        for (size_t j = 0; j < l->out_dim; j++)
        {
            w[j] = (double *)malloc(l->in_dim * sizeof(double));

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


void initialize_mlp_glorot_normal_(const unsigned int seed, mlp *model)
{
    srand(seed);

    double **w;
    double *b;
    layer *l;

    for (size_t i = 0; i < model->n_layers; i++)
    {   
        l = model->layers[i];
        w = (double **) malloc(l->out_dim * sizeof(double *));
        b = (double *) calloc(l->out_dim, sizeof(double));
        for (size_t j = 0; j < l->out_dim; j++)
        {
            w[j] = (double *)malloc(l->in_dim * sizeof(double));

            for (size_t k = 0; k < l->in_dim; k++)
            {
                w[j][k] = glorot_random_normal(l->in_dim, l->out_dim);
            }
            b[j] = glorot_random_normal(l->in_dim, l->out_dim);
        }
        l->w = w;
        l->b = b;
    }
}
