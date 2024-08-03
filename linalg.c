#include "linalg.h"


double *vector_addition(double *v1, double *v2, const size_t dim)
{
    double *result = calloc(dim, sizeof(double));

    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v1[i] + v2[i];
    }
    return result;
}


double *vector_subtraction(double *v1, double *v2, const size_t dim)
{
    double *result = calloc(dim, sizeof(double));
    
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v1[i] - v2[i];
    }
    return result;
}


double vector_sum(double *v, const size_t dim)
{
    double sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += v[i];
    }
    return sum;
}


double *vector_scalar_product(double *v, double s, const size_t dim)
{
    double *result = calloc(dim, sizeof(double));
    
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v[i] * s;
    }
    return result;
}


double *vector_scalar_addition(double *v, double s, const size_t dim)
{
    double *result = calloc(dim, sizeof(double));
    
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v[i] + s;
    }
    return result;
}


double *vector_elementwise_product(double *v1, double *v2, const size_t dim)
{
    double *result  = calloc(dim, sizeof(double));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v1[i] * v2[i];
    }
    return result;
}


double vector_dot_product(double *v1, double *v2, const size_t dim)
{
    double result = 0.0;
    for (size_t i = 0; i < dim; i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}


double **vector_outer_product(double *col, double *row, const size_t col_dim,
        const size_t row_dim)
{
    double **matrix = malloc(col_dim * sizeof(double *));

    for (size_t i = 0; i < col_dim; i++)
    {
        matrix[i] = calloc(row_dim, sizeof(double));
        for (size_t j = 0; j < row_dim; j++)
        {
            matrix[i][j] = col[i] * row[j];
        }
    }
    return matrix;
}


double *matrix_vector_dot_product(double **m, double *v, const size_t n_rows, 
        const size_t n_cols)
{
    double *result = calloc(n_rows, sizeof(double));

    for (size_t r = 0; r < n_rows; r++)
    {
        for (size_t c = 0; c < n_cols; c++)
        {
            result[r] += m[r][c] * v[c];
        }
    }
    return result;
}


double **matmul(double **m1, double **m2, const size_t n_rows_m1,
        const size_t n_cols_m1, const size_t n_cols_m2)
{
    double **result;
    result = malloc(n_rows_m1 * sizeof(double *));

    for (size_t i = 0; i < n_rows_m1; i++)
    {
        result[i] = calloc(n_cols_m2, sizeof(double));
        for (size_t j = 0; j < n_cols_m1; j++)
        {
            for (size_t k = 0; k < n_cols_m2; k++)
            {
                result[i][k] += m1[i][j] * m2[j][k];
            }
        }
    }
    return result;
}


double **matrix_transpose(double **m, const size_t n_rows, const size_t n_cols)
{
    double **transposed_matrix = malloc(n_cols * sizeof(double *));
    for (size_t i = 0; i < n_cols; i++)
    {
        transposed_matrix[i] = calloc(n_rows, sizeof(double));
        for (size_t j = 0; j < n_rows; j++)
        {
            transposed_matrix[i][j] = m[j][i];
        }
    }
    return transposed_matrix;
}


double **matrix_scalar_product(double **m, double s, const size_t n_rows,
        const size_t n_cols)
{
    double **result = malloc(n_rows * sizeof(double *));
    for (size_t i = 0; i < n_rows; i++)
    {
        result[i] = vector_scalar_product(m[i], s, n_cols);
    }
    return result;
}


double **matrix_scalar_addition(double **m, double s, const size_t n_rows,
        const size_t n_cols)
{
    double **result = malloc(n_rows * sizeof(double *));
    for (size_t i = 0; i < n_rows; i++)
    {
        result[i] = vector_scalar_addition(m[i], s, n_cols);
    }
    return result;
}


double mean(double *v, const size_t dim)
{
    double sum = vector_sum(v, dim);
    return sum / (double) dim;
}
