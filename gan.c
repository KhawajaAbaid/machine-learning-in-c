#include "linalg.h"
#include "datasets.h"
#include "mlp_common.h"
#include <stdio.h>
#include <time.h>
#include <stdbool.h>



typedef struct
{
    double **z_all;
    double **a_all;

} outputs_with_logits;


static inline double binary_crossentropy_loss(double y, double y_pred)
{
    return -((y * log(y_pred)) + ((1.0f - y) * log(1.0f - y_pred)));
}


static inline double *leaky_relu(double *x, double negative_slope, 
        const size_t dim)
{
    double *result = malloc(dim * sizeof(double));
    double r;
    for (size_t i = 0; i < dim; i++)
    {
        r = (x[i] > 0.0) ? x[i] : 0.0;
        r += negative_slope * ((x[i] < 0) ? x[i] : 0.0);
        result[i] = r;
    }
    return result;
}


static inline double *leaky_relu_prime(double *x, double negative_slope, 
        const size_t dim)
{
    double *result = malloc(dim * sizeof(double));
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
    double *result = malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = tanhf(x[i]);
    }
    return result;
}


static inline double square(double x)
{
    return x * x;
}


static inline double *tanh_prime(double *x, const size_t dim)
{
    double *result = malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = 1.0f - square(tanh(x[i]));
    }
    return result;
}


static inline double *sample_noise(const unsigned int seed, size_t dim)
{
    srand(seed);
    double *noise = (double *) malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++)
    {
        noise[i] = random_normal();
    }
    return noise;
}


static inline double *generator_forward_pass(mlp *model, double *x)
{
    double *a; // activations
    double *z; // logits
    double *temp;
    a = x;
    size_t last_layer_index = model->n_layers - 1;
    for (size_t i = 0; i < model->n_layers; i++)
    {
        z = matrix_vector_dot_product(model->layers[i]->w, a,
                model->layers[i]->out_dim, model->layers[i]->in_dim);
        if (a != x)
        {
            free(a);
        }
        temp = z;
        z = vector_addition(z, model->layers[i]->b,
                model->layers[i]->out_dim);
        free(temp); 
        if (i != last_layer_index)
        {
            a = leaky_relu(z, 0.2, model->layers[i]->out_dim);
        }
        else
        {
            a = tanh_vectorized(z, model->layers[i]->out_dim);
        }
        free(z);
    }
    return a;
}



static inline double *discriminator_forward_pass(mlp *model, double *x)
{
    double *a; // activations
    double *z; // logits
    double *temp;
    a = x;
    size_t last_layer_index = model->n_layers - 1;
    for (size_t i = 0; i < model->n_layers; i++)
    {
        z = matrix_vector_dot_product(model->layers[i]->w, a,
                model->layers[i]->out_dim, model->layers[i]->in_dim);
        if (a != x)
        {
            free(a);
        }
        temp = z;
        z = vector_addition(z, model->layers[i]->b,
                model->layers[i]->out_dim);
        free(temp); 
        if (i != last_layer_index)
        {
            a = leaky_relu(z, 0.2, model->layers[i]->out_dim);
        }
        else
        {
            a = sigmoid(z, model->layers[i]->out_dim);
        }
        free(z);
    }
    return a;
}


static inline outputs_with_logits *generator_forward_pass_with_logits(
        mlp *model, double *x)
{
    double *a; // activations
    double *z; // logits
    double *temp;
    double **z_all = malloc(model->n_layers * sizeof(double *));
    double **a_all = malloc((model->n_layers + 1) * sizeof(double *));

    size_t last_layer_index = model->n_layers - 1;

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
        if (i != last_layer_index)
        {
            a = leaky_relu(z, 0.2, model->layers[i]->out_dim);
        }
        else
        {
            a = tanh_vectorized(z, model->layers[i]->out_dim);
        }
        z_all[i] = z;
        a_all[i + 1] = a;
    }

    outputs_with_logits *a_and_z = malloc(sizeof(outputs_with_logits));
    a_and_z->z_all = z_all;
    a_and_z->a_all = a_all;

    return a_and_z;
}


static inline outputs_with_logits *discriminator_forward_pass_with_logits(
        mlp *model, double *x)
{
    double *a; // activations
    double *z; // logits
    double *temp;
    double **z_all = malloc(model->n_layers * sizeof(double *));
    double **a_all = malloc((model->n_layers + 1) * sizeof(double *));

    size_t last_layer_index = model->n_layers - 1;

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
        if (i != last_layer_index)
        {
            a = leaky_relu(z, 0.2, model->layers[i]->out_dim);
        }
        else
        {
            a = sigmoid(z, model->layers[i]->out_dim);
        }
        z_all[i] = z;
        a_all[i + 1] = a;
    }

    outputs_with_logits *a_and_z = malloc(sizeof(outputs_with_logits));
    a_and_z->z_all = z_all;
    a_and_z->a_all = a_all;

    return a_and_z;
}


grad_and_metrics *discriminator_backprop(mlp *discriminator, double *x, double y)
{
    struct metrics *metrics = (struct metrics *) malloc(sizeof(struct metrics));
    gradient *grad = copy_mlp(discriminator);

    // Forward pass
    double **z_all, **a_all;
    outputs_with_logits *a_and_z = discriminator_forward_pass_with_logits(
            discriminator, x);
    a_all = a_and_z->a_all;
    z_all = a_and_z->z_all;
    free(a_and_z);

    size_t last_layer_index = discriminator->n_layers - 1;
    
    // Compute loss
    metrics->loss = binary_crossentropy_loss(y, a_all[last_layer_index + 1][0]);

    // Compute gradients 
    double **dloss_dz_all = malloc(discriminator->n_layers * sizeof(double *));
    dloss_dz_all[last_layer_index] = vector_subtraction(
            a_all[last_layer_index + 1], &y, 1);

    grad->layers[last_layer_index]->w = vector_outer_product(
            dloss_dz_all[last_layer_index], a_all[last_layer_index],
            discriminator->layers[last_layer_index]->out_dim,
            discriminator->layers[last_layer_index]->in_dim);

    grad->layers[last_layer_index]->b = dloss_dz_all[last_layer_index];

    // Now iteratively compute grads
    double **w_t;
    double *dloss_da;
    double *da_dz;
    
    for (int i = discriminator->n_layers - 2; i >= 0; i--)
    {
        w_t = matrix_transpose(discriminator->layers[i + 1]->w,
                discriminator->layers[i + 1]->out_dim,
                discriminator->layers[i + 1]->in_dim);
        dloss_da = matrix_vector_dot_product(w_t,
                dloss_dz_all[i + 1],
                discriminator->layers[i + 1]->in_dim,
                discriminator->layers[i + 1]->out_dim);
        free_2d_matrix(w_t, discriminator->layers[i]->out_dim);
        da_dz = leaky_relu_prime(z_all[i], 0.2,
                discriminator->layers[i]->out_dim);
        dloss_dz_all[i] = vector_elementwise_product(da_dz, dloss_da,
                discriminator->layers[i]->out_dim);
        free(dloss_da);
        free(da_dz);
        grad->layers[i]->w = vector_outer_product(
                dloss_dz_all[i], a_all[i],
                discriminator->layers[i]->out_dim,
                discriminator->layers[i]->in_dim);
        grad->layers[i]->b = dloss_dz_all[i];
    }

    grad_and_metrics *results = malloc(sizeof(grad_and_metrics));
    results->grad = grad;
    
    metrics->accuracy = a_all[last_layer_index + 1][0] >= 0.5f;
    results->metrics = metrics;

    // Free mem
    for (size_t i = 0; i < discriminator->n_layers; i++)
    {
        free(z_all[i]);
        // activations contains 1 more elem than z
        // and 0th index points to inputs which we don't want to free (yet)
        free(a_all[i + 1]);
    }
    free(z_all);
    free(a_all);
    free(dloss_dz_all);

    return results;
}



// noise is a 100 - or whatever the input dim of generator - dim vector
grad_and_metrics *generator_backprop(mlp *generator, mlp *discriminator,
        double *noise)
{
    double y = 1.0f; 
    struct metrics *metrics = (struct metrics *) malloc(sizeof(struct metrics));
    // Generator Forward pass
    double **gen_z_all, **gen_a_all;
    outputs_with_logits *a_and_z = generator_forward_pass_with_logits(
            generator, noise);
    gen_z_all = a_and_z->z_all;
    gen_a_all = a_and_z->a_all;
    free(a_and_z);

    // Discriminator forward
    double **disc_z_all, **disc_a_all;
    a_and_z = discriminator_forward_pass_with_logits(discriminator, 
            gen_a_all[generator->n_layers]);
    disc_z_all = a_and_z->z_all;
    disc_a_all = a_and_z->a_all;
    free(a_and_z);

    // Compute loss
    double loss = binary_crossentropy_loss(y,
            disc_a_all[discriminator->n_layers][0]);

    // We only need the gradient for the generator but we have to work backward
    // throught the discriminator first.
    // We won't store any intermediate values, rather we're only concerned with
    // the gradient of discriminator's loss w.r.t. the logits of its input 
    // layer as those logits are based on the output activations of the
    // generator (i.e. the fake image).
    size_t last_layer_index = discriminator->n_layers - 1;
    double *disc_dloss_dz_last = vector_subtraction(
            disc_a_all[last_layer_index + 1], &y, 1);
    double *temp;
    // Now iteratively compute grads
    double **w_t;
    double *dloss_da;
    double *da_dz;
    
    for (int i = discriminator->n_layers - 2; i >= 0; i--)
    {
        w_t = matrix_transpose(discriminator->layers[i + 1]->w,
                discriminator->layers[i + 1]->out_dim,
                discriminator->layers[i + 1]->in_dim);
        dloss_da = matrix_vector_dot_product(w_t,
                disc_dloss_dz_last,
                discriminator->layers[i + 1]->in_dim,
                discriminator->layers[i + 1]->out_dim);
        free_2d_matrix(w_t, discriminator->layers[i]->out_dim);
        da_dz = leaky_relu_prime(disc_z_all[i], 0.2,
                discriminator->layers[i]->out_dim);
        temp = disc_dloss_dz_last;
        disc_dloss_dz_last = vector_elementwise_product(da_dz, dloss_da,
                discriminator->layers[i]->out_dim);
        free(temp);
        free(dloss_da);
        free(da_dz);
    }

    // Now we compute and store gradients for the generator
    // Since now we have the last dloss_dz given by the discriminator
    gradient *grad = copy_mlp(generator);
    double **dloss_dz_all = malloc(generator->n_layers * sizeof(double *));
    last_layer_index = generator->n_layers - 1;
    w_t = matrix_transpose(discriminator->layers[0]->w,
            discriminator->layers[0]->out_dim,
            discriminator->layers[0]->in_dim);
    dloss_da = matrix_vector_dot_product(w_t,
            disc_dloss_dz_last,
            discriminator->layers[0]->in_dim,
            discriminator->layers[0]->out_dim);
    free(disc_dloss_dz_last);
    free_2d_matrix(w_t, discriminator->layers[0]->in_dim);
    // or generator->layers[last_layer_index]->out_dim
    da_dz = tanh_prime(gen_z_all[last_layer_index],
            generator->layers[last_layer_index]->out_dim);
    dloss_dz_all[last_layer_index] = vector_elementwise_product(
            da_dz, dloss_da, generator->layers[last_layer_index]->out_dim);
    free(da_dz);
    free(dloss_da);
    grad->layers[last_layer_index]->w = vector_outer_product(
            dloss_dz_all[last_layer_index],
            gen_a_all[last_layer_index],
            generator->layers[last_layer_index]->out_dim,
            generator->layers[last_layer_index]->in_dim);
    grad->layers[last_layer_index]->b = dloss_dz_all[last_layer_index];

    for (int i = generator->n_layers - 2; i >=0; i--)
    {
        w_t = matrix_transpose(generator->layers[i + 1]->w,
                generator->layers[i + 1]->out_dim,
                generator->layers[i + 1]->in_dim);
        dloss_da = matrix_vector_dot_product(w_t,
                dloss_dz_all[i + 1],
                generator->layers[i + 1]->in_dim,
                generator->layers[i + 1]->out_dim);
        free_2d_matrix(w_t, generator->layers[i + 1]->in_dim);
        da_dz = leaky_relu_prime(gen_z_all[i], 0.2,
                generator->layers[i]->out_dim);
        dloss_dz_all[i] = vector_elementwise_product(dloss_da, da_dz,
                generator->layers[i]->out_dim);
        free(dloss_da);
        free(da_dz);
        grad->layers[i]->w = vector_outer_product(
                dloss_dz_all[i], gen_a_all[i],
                generator->layers[i]->out_dim,
                generator->layers[i]->in_dim);
        grad->layers[i]->b = dloss_dz_all[i];
    }

    grad_and_metrics *results = malloc(sizeof(grad_and_metrics));
    results->grad = grad;
    
    metrics->loss = loss;
    results->metrics = metrics;

    // Free mem
    for (size_t i = 0; i < generator->n_layers; i++)
    {
        free(gen_z_all[i]);
        free(gen_a_all[i]);
        free(disc_a_all[i + 1]);
        free(disc_z_all[i]);
    }
    free(gen_a_all[generator->n_layers]);
    free(gen_z_all);
    free(gen_a_all);
    free(disc_a_all);
    free(disc_z_all);
    free(dloss_dz_all);

    return results;
}


struct metrics *discriminator_train_step(mlp *discriminator,
        mlp *generator, double *x, double learning_rate, 
        const size_t batch_size)
{
    struct metrics *metrics = malloc(sizeof(struct metrics));
    metrics->loss = 0.0;
    metrics->accuracy = 0.0;

    size_t latent_dim = generator->layers[0]->in_dim;
    size_t image_dim = discriminator->layers[0]->in_dim;

    mlp *accumulated_gradient = copy_mlp(discriminator);
    init_with_zeros(accumulated_gradient);

    // Train on real images
    grad_and_metrics *results; 
    for (size_t i = 0; i < batch_size; i++)
    {
        results = discriminator_backprop(discriminator, x + (i * image_dim),
                1.0f);
        accumulate_grad(accumulated_gradient, results->grad);
        metrics->loss += results->metrics->loss;
        metrics->accuracy += results->metrics->accuracy;
        free_grad_and_metrics(results);
    }

    // Train on fake images
    double *fake_image;
    double *noise;
    for (size_t i = 0; i < batch_size; i++)
    {
        noise = sample_noise(time(NULL) + i, latent_dim);
        fake_image = generator_forward_pass(generator, noise);
        free(noise);
        results = discriminator_backprop(discriminator, fake_image, 0.0);
        free(fake_image);
        accumulate_grad(accumulated_gradient, results->grad);
        metrics->loss += results->metrics->loss;
        metrics->accuracy += results->metrics->accuracy;
        free_grad_and_metrics(results);
    }

    divide_grad_by_batch_size(accumulated_gradient, batch_size);
    metrics->loss /= batch_size; // Average loss
    metrics->accuracy /= batch_size * 2;
    
    update_weights(discriminator, accumulated_gradient, learning_rate);
    free_gradient(accumulated_gradient);
    return metrics;
}


struct metrics *generator_train_step(mlp *generator, mlp *discriminator,
        double learning_rate, const size_t batch_size)
{
    struct metrics *metrics = malloc(sizeof(struct metrics));
    metrics->loss = 0.0;

    size_t latent_dim = generator->layers[0]->in_dim;
    size_t image_dim = discriminator->layers[0]->in_dim;

    mlp *accumulated_gradient = copy_mlp(generator);
    // Init accumulated gradient values to zero
    init_with_zeros(accumulated_gradient);

    double *noise;
    grad_and_metrics *results;
    for (size_t i = 0; i < batch_size; i++)
    {
        noise = sample_noise(time(NULL) + i * (i + 1), latent_dim);
        results = generator_backprop(generator, discriminator, noise);
        accumulate_grad(accumulated_gradient, results->grad);
        metrics->loss += results->metrics->loss;
        free_grad_and_metrics(results);
    }
    divide_grad_by_batch_size(accumulated_gradient, batch_size);
    metrics->loss /= batch_size;

    update_weights(generator, accumulated_gradient, learning_rate);
    free_gradient(accumulated_gradient);
    return metrics;
}


// Thanks ChatGPT
static inline void save_array_as_pgm(const char *filename, int width, 
        int height, unsigned char *array)
{
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Could not open file");
        exit(EXIT_FAILURE);
    }

    // Write the PGM header
    fprintf(file, "P5\n%d %d\n255\n", width, height);

    // Write the pixel data
    fwrite(array, sizeof(unsigned char), width * height, file);

    fclose(file);
}


static inline void generate_and_save_image(mlp *generator, const char *filename)
{
    double *noise = sample_noise(time(NULL), generator->layers[0]->in_dim);
    double *image_f = generator_forward_pass(generator, noise);
    free(noise);
    
    size_t image_dim = generator->layers[generator->n_layers - 1]->out_dim;
    // Scale it back up
    unsigned char *image_uc = malloc(image_dim);
    for (size_t i = 0; i < image_dim; i++)
    {
        image_uc[i] = (unsigned char) (127.5 * image_f[i] + 127.5);
    }
    save_array_as_pgm(filename, 28, 28, image_uc);
    free(image_uc);
    free(image_f);
}


void fit(mlp *generator, mlp *discriminator, dataset *ds, double learning_rate,
        const size_t batch_size, const size_t n_epochs)
{
    size_t n_batches = ds->n_samples / batch_size;   // ignore the `remainder` samples 
    struct metrics *gen_batch_metrics;
    struct metrics *gen_epoch_metrics = malloc(sizeof(struct metrics));
    struct metrics *disc_batch_metrics;
    struct metrics *disc_epoch_metrics = malloc(sizeof(struct metrics));

    char filename[50];
    time_t now;
    struct tm *local;

    size_t image_dims = discriminator->layers[0]->in_dim;
    // 1. Training loop
    for (size_t epoch = 0; epoch < n_epochs; epoch++)
    {
        gen_epoch_metrics->loss = 0.0;
        disc_epoch_metrics->loss = 0.0;
        disc_epoch_metrics->accuracy = 0.0;
        // 2. Create batches
        size_t batch;
        for (batch = 0; batch < n_batches; batch++)
        {
            // 3. Pass each batch to train_step
            disc_batch_metrics = discriminator_train_step(discriminator,
                    generator, ds->x + (batch * batch_size * image_dims),
                    learning_rate, batch_size);
            gen_batch_metrics = generator_train_step(generator, discriminator,
                    learning_rate, batch_size);
            // Save an image after each 25 batches
            if (batch % 25 == 0)
            {
                //time(&now);
                //local = localtime(&now);
                sprintf(filename, "./images/image_epoch_%ld_batch_%ld.pgm",
                        epoch, batch);
                generate_and_save_image(generator, filename);
            }
            // 4. Print logs
            printf("\rEpoch %-4ld | Batch %-4ld | Gen Loss: %-8.5f | Disc Loss: %-8.5f | Disc Acc: %-8.5f",
                    epoch, batch, gen_batch_metrics->loss,
                    disc_batch_metrics->loss, disc_batch_metrics->accuracy);
            gen_epoch_metrics->loss += gen_batch_metrics->loss;
            disc_epoch_metrics->loss += disc_batch_metrics->loss;
            disc_epoch_metrics->accuracy += disc_batch_metrics->accuracy;
            free(gen_batch_metrics);
            free(disc_batch_metrics);
        }
        gen_epoch_metrics->loss /= n_batches;
        disc_epoch_metrics->loss /= n_batches;
        disc_epoch_metrics->accuracy /= n_batches;
        printf("\rEpoch %-4ld | Batch %-4ld | Gen Loss: %-8.5f | Disc Loss: %-8.5f | Disc Acc: %-8.5f",
                epoch, batch, gen_epoch_metrics->loss,
                disc_epoch_metrics->loss, disc_epoch_metrics->accuracy);
        printf("\n");
    }
    free(gen_epoch_metrics);
    free(disc_epoch_metrics);
}


void main()
{
    size_t generator_dims[4] = {100, 256, 512, 784};
    
    printf("Creating generator...\n");
    int num_layers = 3;
    mlp *generator = create_mlp(num_layers, generator_dims);
    
    printf("Initializing weights...\n");
    initialize_weights_glorot_normal(8888, generator);

    size_t discriminator_dims[4] = {784, 256, 128, 1};
    
    printf("Creating discriminator...\n");
    num_layers = 3;
    mlp *discriminator = create_mlp(num_layers, discriminator_dims);
    
    printf("Initializing weights...\n");
    initialize_weights_glorot_normal(7564, discriminator);
    
    printf("Loading mnist dataset...\n");
    dataset *mnist_ds = load_mnist(true);

    printf("Training...\n");
    fit(generator, discriminator, mnist_ds, 0.01, 512, 100);

    free_mlp(generator);
    free_mlp(discriminator);
    free_dataset(mnist_ds);
}
