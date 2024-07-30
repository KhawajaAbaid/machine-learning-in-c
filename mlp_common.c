#include "mlp_common.h"


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


void initialize_weights(const unsigned int seed, mlp *model,
        float (*initializer)(void))
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
            w[j] = (float *)malloc(l->in_dim * sizeof(float));

            for (size_t k = 0; k < l->in_dim; k++)
            {
                w[j][k] = initializer();
            }
            b[j] = initializer();
        }
        l->w = w;
        l->b = b;
    }
}

float *sigmoid(float *x, const size_t dim)
{
    float *result = (float *) calloc(dim, sizeof(float));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return result;
}


float *sigmoid_prime(float *x, const size_t dim)
{
    float *result = (float *) malloc(dim * sizeof(float));

    float *sig = sigmoid(x, dim);
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = sig[i] * (1.0f - sig[i]); 
    }
    return result;
}


float crossentropy_loss(float *y, float *y_pred, const size_t dim)
{
    float loss = 0.0f;

    for (size_t i = 0; i < dim; i++)
    {
        loss += y[i] * log(y_pred[i]) + (1.0 - y[i]) * log(1.0 - y_pred[i]); 
    }
    return -loss;
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


// Divide sum of individual instance gradients by batch size to get avg
void divide_grad_by_batch_size(mlp *accumulated_gradient, size_t batch_size)
{
    for (size_t i = 0; i < accumulated_gradient->n_layers; i++)
    {
        for (size_t j = 0; j < accumulated_gradient->layers[i]->out_dim; j++)
        {
            for (size_t k = 0; k < accumulated_gradient->layers[i]->in_dim; k++)
            {
                accumulated_gradient->layers[i]->w[j][k] /= (float) batch_size;
            }
            accumulated_gradient->layers[i]->b[j] /= (float) batch_size;
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


void init_with_zeros(mlp *model)
{
    // Init accumulated gradient values to zero
    for (size_t i = 0; i < model->n_layers; i++)
    {
        model->layers[i]->w = (float **) malloc(
                model->layers[i]->out_dim * sizeof(float *));
        for (size_t j = 0; j < model->layers[i]->out_dim; j++)
        {
            model->layers[i]->w[j] = (float *) calloc(
                    model->layers[i]->in_dim,
                    sizeof(float));
        }
        model->layers[i]->b = (float *) calloc(
                model->layers[i]->out_dim,
                sizeof(float));
    }
}


void initialize_weights_glorot_normal(const unsigned int seed, mlp *model)
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
            w[j] = (float *)malloc(l->in_dim * sizeof(float));

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
