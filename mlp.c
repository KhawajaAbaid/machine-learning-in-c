#include "linalg.h"
#include "datasets.h"
#include "random.h"
#include <math.h>


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
    mlp *model = malloc(sizeof(mlp));
    model->layers = malloc(n_layers * sizeof(layer *));
    model->n_layers = n_layers;

    for (size_t i = 0; i < n_layers; i++)
    {
        model->layers[i] = malloc(sizeof(layer));
        model->layers[i]->in_dim = dims[i];
        model->layers[i]->out_dim = dims[i + 1];
    }

    return model;
}


// Creates copy of a given mlp of the same structure but without weights
mlp *copy_mlp(mlp *source)
{
    mlp *copied = malloc(sizeof(mlp));
    copied->layers = malloc(source->n_layers * sizeof(layer *));
    copied->n_layers = source->n_layers;
    // set copied layers dims
    for (size_t i = 0; i < source->n_layers; i++)
    {
        copied->layers[i] = malloc(sizeof(layer));
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
        w = malloc(l->out_dim * sizeof(float *));
        b = calloc(l->out_dim, sizeof(float));
        for (size_t j = 0; j < l->out_dim; j++)
        {
            w[j] = malloc(l->in_dim * sizeof(float));

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


float *sigmoid(float *x, const size_t dim)
{
    float *result = calloc(dim, sizeof(float));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return result;
}


float *sigmoid_prime(float *x, const size_t dim)
{
    float *result = malloc(dim * sizeof(float));

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


size_t argmax(float *x, const size_t dim)
{
    size_t max_idx = 0;
    float max = 0.0f;

    for (size_t i = 0; i < dim; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}


// FOR A SINGLE INSTANCE runs the forward pass computes grads
grad_and_metrics *backprop(mlp *model, float *x, float *y)
{
    struct metrics *metrics = (struct metrics *) malloc(sizeof(struct metrics *));
    gradient *grad = copy_mlp(model);
    
    // Forward pass
    float *a; // activations
    float *z; // logits
    float *temp;
    float **z_all = malloc(model->n_layers * sizeof(float *));
    float **a_all = malloc((model->n_layers + 1) * sizeof(float *));
    float **dloss_dz_all = malloc(model->n_layers * sizeof(float *));

    // Okay please forgive me for this but in a_all array, i'm reserving the
    // index 0 for inputs. It breaks one to one correspondence with every other
    // relevant array but idk what else to do here
    a = x;
    a_all[0] = a;
    for (size_t i = 0; i < model->n_layers; i++)
    {
        z = matrix_vector_dot_product(model->layers[i]->w, a,
                model->layers[i]->out_dim, model->layers[i]->in_dim);
        temp = z;
        z = vector_addition(z, model->layers[i]->b,
                model->layers[i]->out_dim);
        free(temp);
        a = sigmoid(z, model->layers[i]->out_dim);
        z_all[i] = z;
        a_all[i + 1] = a;
    }

    size_t last_layer_index = model->n_layers - 1;

    // Compute loss
    metrics->loss = crossentropy_loss(y, a_all[last_layer_index + 1],
            model->layers[last_layer_index]->out_dim);

    // FUN PART: LETS COMPUTE GRADIENTS!!!
    dloss_dz_all[last_layer_index] = vector_subtraction(
            a_all[last_layer_index + 1], y,
            model->layers[last_layer_index]->out_dim);
    
    grad->layers[last_layer_index]->w = vector_outer_product(
            dloss_dz_all[last_layer_index], a_all[last_layer_index],
            model->layers[last_layer_index]->out_dim,
            model->layers[last_layer_index]->in_dim);
    
    grad->layers[last_layer_index]->b = dloss_dz_all[last_layer_index];

    // Now iteratively compute gradients for logits of each layer, backwards
    float **w_t;
    float *dloss_da;
    float *da_dz;
    for (int i = model->n_layers - 2; i >= 0; i--)
    {
        w_t = matrix_transpose(model->layers[i + 1]->w,
                model->layers[i + 1]->out_dim, model->layers[i + 1]->in_dim);
        dloss_da = matrix_vector_dot_product(
                w_t,
                dloss_dz_all[i + 1],
                model->layers[i + 1]->in_dim,
                model->layers[i + 1]->out_dim);
        free_2d_matrix(w_t, model->layers[i]->out_dim);
        da_dz = sigmoid_prime(z_all[i], model->layers[i]->out_dim);
        dloss_dz_all[i] = vector_elementwise_product(da_dz, dloss_da,
                model->layers[i]->out_dim);
        free(dloss_da);
        free(da_dz);
        grad->layers[i]->w = vector_outer_product(
                dloss_dz_all[i], a_all[i], // Remember: a_all[i] means the activations of the previous layer
                model->layers[i]->out_dim, model->layers[i]->in_dim
                );
        grad->layers[i]->b = dloss_dz_all[i]; 
    }
    grad_and_metrics *results = malloc(sizeof(grad_and_metrics));
    results->grad = grad;
    
    int label_pred = argmax(a_all[last_layer_index + 1],
            model->layers[last_layer_index]->out_dim);
    int actual_label = argmax(y, model->layers[last_layer_index]->out_dim); 
    metrics->accuracy = label_pred == actual_label;
    
    results->metrics = metrics;

    // Free mem
    for (size_t i = 0; i < model->n_layers; i++)
    {
        free(z_all[i]);
        // activations contains 1 more elem than z
        // and 0th index posize_ts to inputs which we don't want to free yet
        free(a_all[i + 1]);
    }
    free(z_all);
    free(a_all);
    free(dloss_dz_all);

    return results;
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


struct metrics *train_step(mlp *model, float *x_b, float *y_b,
        float learning_rate, const size_t batch_size) 
{
    struct metrics *metrics = (struct metrics *) malloc(sizeof(struct metrics));
    metrics->loss = 0.0f;
    metrics->accuracy = 0.0f;

    size_t image_dims = model->layers[0]->in_dim;
    size_t num_classes = model->layers[model->n_layers - 1]->out_dim;
    mlp *accumulated_gradient = copy_mlp(model);
    // Init accumulated gradient values to zero
    for (size_t i = 0; i < model->n_layers; i++)
    {
        accumulated_gradient->layers[i]->w = malloc(
                accumulated_gradient->layers[i]->out_dim * sizeof(float *));
        for (size_t j = 0; j < accumulated_gradient->layers[i]->out_dim; j++)
        {
            accumulated_gradient->layers[i]->w[j] = calloc(
                    accumulated_gradient->layers[i]->in_dim,
                    sizeof(float));
        }
        accumulated_gradient->layers[i]->b = calloc(
                accumulated_gradient->layers[i]->out_dim,
                sizeof(float));
    }

    grad_and_metrics *results;
    for (size_t i = 0; i < batch_size; i++)
    {
        results = backprop(model, x_b + (i * image_dims),
                y_b + (i * num_classes));
        accumulate_grad(accumulated_gradient, results->grad);
        metrics->loss += results->metrics->loss;
        metrics->accuracy += results->metrics->accuracy;
        free_grad_and_metrics(results);
    }


    metrics->loss /= batch_size; // Average loss
    metrics->accuracy /= batch_size;
    
    update_weights(model, accumulated_gradient, learning_rate);
    free_mlp(accumulated_gradient);
    return metrics;
}


void fit(mlp *model, dataset *ds, float learning_rate, const size_t batch_size,
        const size_t n_epochs)
{
    size_t n_batches = ds->n_samples / batch_size;   // ignore the `remainder` samples 
    struct metrics *batch_metrics;
    struct metrics *epoch_metrics = malloc(sizeof(struct metrics));
    
    size_t image_dims = model->layers[0]->in_dim;
    size_t num_classes = model->layers[model->n_layers - 1]->out_dim;
    // 1. Training loop
    for (size_t epoch = 0; epoch < n_epochs; epoch++)
    {
        epoch_metrics->loss = 0.0f;
        epoch_metrics->accuracy = 0.0f;
        // 2. Create batches
        size_t batch;
        for (batch = 0; batch < n_batches; batch++)
        {
            // 3. Pass each batch to train_step
            batch_metrics = train_step(model, 
                    ds->x + (batch * batch_size * image_dims),
                    ds->y + (batch * batch_size * num_classes), learning_rate,
                    batch_size);
            // 4. Prsize_t logs
            printf("\rEpoch %-4ld | Batch %-4ld | Loss: %-8.5f | Accuracy: %-8.5f",
                    epoch, batch, batch_metrics->loss, batch_metrics->accuracy);
            epoch_metrics->loss += batch_metrics->loss;
            epoch_metrics->accuracy += batch_metrics->accuracy;
            free(batch_metrics);
        }
        epoch_metrics->loss /= n_batches;
        epoch_metrics->accuracy /= n_batches;
        printf("\rEpoch %-4ld | Batch %-4ld | Loss: %-8.5f | Accuracy: %-8.5f",
                epoch, batch, epoch_metrics->loss, epoch_metrics->accuracy);
        printf("\n");
    }
}


void main()
{
    printf("\n\t\t==================================\n");
    printf("\t\t||\t  mlp.c \t\t||\n");
    printf("\t\t==================================\n");
   
    printf("Loading data....\n");
    dataset *mnist_ds = load_mnist();

    // let's create a 3 layer model
    printf("Creating model....\n");
    size_t dims[4] = {784, 128, 64, 10};
    mlp *model = create_mlp(3, dims);
 
    printf("Initializing weights...\n");
    initialize_weights(1001, model);

    printf("Training...\n");
    fit(model, mnist_ds, 0.005f, 512, 10);

    free_mlp(model);
    free(mnist_ds->x);
    free(mnist_ds->y);
    free(mnist_ds);}
