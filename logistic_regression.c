/*
 * Logistic regression for binary classificaiton.
 */
#include "linalg.h"
#include "random.h"
#include "datasets.h"
#include <stdio.h>
#include <math.h>


typedef struct 
{
    double *w;
    double b;
    size_t in_dim;
    size_t out_dim;    
} model;

typedef model gradient;

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


static inline void free_model(model *m)
{
    free(m->w);
    free(m);
}


static inline void free_gradient(gradient *grad)
{
    free_model(grad);
}


static inline void free_grad_and_metrics(grad_and_metrics *gam)
{
    free_model(gam->grad);
    free(gam);
}


void initialize_weights_(const unsigned int seed, model *m)
{
    srand(seed);
    
    m->w = malloc(m->in_dim * sizeof(double));

    for (size_t i = 0; i < m->in_dim; i++)
    {
       m->w[i] = random_normal();
    }
    m->b = random_normal();
}


double sigmoid(double x)
{
    return 1.0f / (1.0f + expf(-x));
}


double binary_crossentropy_loss(double y, double y_pred)
{
    return -((y * logf(y_pred)) + ((1.0f - y) * logf(1.0f - y_pred)));
}


// Runs backpropagation for a single instance
grad_and_metrics *backprop(model *m, double *x, double y)
{
    struct metrics *metrics = (struct metrics *) malloc(sizeof(struct metrics *));
    gradient *grad = (gradient *) malloc(sizeof(gradient));
    grad->in_dim = m->in_dim;
    grad->out_dim = m->out_dim;
    
    double z, a;     // z: logits    a: activations
    z = vector_dot_product(m->w, x, m->in_dim);
    z += m->b;
    a = sigmoid(z);

    // Compute loss
    double loss = binary_crossentropy_loss(y, a);

    // Compute gradients
    double dloss_dz = a - y;
    double *grad_loss_wrt_w = vector_scalar_product(x, dloss_dz, m->in_dim);
    double grad_loss_wrt_b = dloss_dz;

    grad->w = grad_loss_wrt_w;
    grad->b = grad_loss_wrt_b;
    
    metrics->loss = loss;
    metrics->accuracy = (y == (a >= 0.5f));

    grad_and_metrics *gam = malloc(sizeof(grad_and_metrics));
    gam->grad = grad;
    gam->metrics = metrics;

    return gam;
}


static inline void accumulate_gradient_(gradient *accum_grad, gradient *new_grad)
{
    for (size_t i = 0; i < accum_grad->in_dim; i++)
    {
        accum_grad->w[i] += new_grad->w[i];
    }
    accum_grad->b += new_grad->b;
}


static inline void divide_grad_by_batch_size_(gradient *accum_grad,
        size_t batch_size)
{
    for (size_t i = 0; i < accum_grad->in_dim; i++)
    {
        accum_grad->w[i] /= (double) batch_size;
    }
    accum_grad->b /= (double) batch_size;
}


static inline void update_weights_(model *m, gradient *grad, double learning_rate)
{
    for (size_t i = 0; i < m->in_dim; i++)
    {
        m->w[i] -= learning_rate * grad->w[i];
    }
    m->b -= learning_rate * grad->b;
}


struct metrics *train_step(model *m, double *x, double *y, double learning_rate,
        const size_t batch_size)
{
    struct metrics *metrics = malloc(sizeof(struct metrics));
    metrics->accuracy = 0.0f;
    metrics->loss = 0.0f;

    // Total gradient for a batch
    model *accum_grad = malloc(sizeof(model));
    accum_grad->in_dim = m->in_dim;
    accum_grad->out_dim = m->out_dim;
    accum_grad->w = calloc(accum_grad->in_dim, sizeof(double));
    accum_grad->b = 0.0f;

    grad_and_metrics *result;
    for (int i = 0; i < batch_size; i++)
    {
        result = backprop(m, x + (i * m->in_dim), y[i]);
        accumulate_gradient_(accum_grad, result->grad);
        free_gradient(result->grad);
        metrics->accuracy += result->metrics->accuracy;
        metrics->loss += result->metrics->loss;
        free(result->metrics);
    }
    free(result);

    divide_grad_by_batch_size_(accum_grad, batch_size);

    // Update weights
    update_weights_(m, accum_grad, learning_rate);

    metrics->accuracy /= batch_size;
    metrics->loss /= batch_size;

    free_gradient(accum_grad);
    return metrics;
}


void fit(model *m, dataset *ds, double learning_rate, const size_t batch_size,
        const size_t n_epochs)
{
    size_t n_batches = ds->n_samples / batch_size;
    
    struct metrics *batch_metrics;
    struct metrics *epoch_metrics = malloc(sizeof(struct metrics));
    size_t batch = 0;
    for (size_t epoch = 0; epoch < n_epochs; epoch++)
    {
        epoch_metrics->loss = 0.0f;
        epoch_metrics->accuracy = 0.0f;
        // 2. Create batches
        for (batch = 0; batch < n_batches; batch++)
        {
            // 3. Pass each batch to train_step
            batch_metrics = train_step(m, ds->x + (batch * batch_size * ds->dim),
                    ds->y + (batch * batch_size), learning_rate,
                    batch_size);
            // 4. Print logs
            printf("\rEpoch %-4ld | Batch %-4ld | Loss: %-6.8f | Acc: %-6.8f",
                    epoch, batch, batch_metrics->loss, batch_metrics->accuracy);
            epoch_metrics->accuracy += batch_metrics->accuracy;
            epoch_metrics->loss += batch_metrics->loss;
            free(batch_metrics);
        }
        epoch_metrics->accuracy /= n_batches;
        epoch_metrics->loss /= n_batches;

        printf("\rEpoch %-4ld | Batch %-4ld | Loss: %-6.8f | Acc: %-6.8f",
                    epoch, batch, epoch_metrics->loss, epoch_metrics->accuracy);
        printf("\n");
    }
}


void main()
{
    printf("Loading data...\n");
    dataset *weather_ds = load_weather();

    printf("Creating model...\n");
    model *logistic_regressor = (model *) malloc(sizeof(model));
    logistic_regressor->in_dim = weather_ds->dim;
    logistic_regressor->out_dim = 1; // just for cosmetic purposes. makes no diff
    
    printf("Initializing weights...\n");
    initialize_weights_(1337, logistic_regressor);
    
    printf("Training...\n");
    fit(logistic_regressor, weather_ds, 0.1f, 512, 500);
    
    free(weather_ds->x);
    free(weather_ds->y);
    free(weather_ds);
}
