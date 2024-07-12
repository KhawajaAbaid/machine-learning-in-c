#include <stdlib.h>
#include "linalg.h"


float *vector_addition(float *v1, float *v2, const size_t dim)
{
    float *result = calloc(dim, sizeof(float));

    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v1[i] + v2[i];
    }
    return result;
}


float *vector_subtraction(float *v1, float *v2, const size_t dim)
{
    float *result = calloc(dim, sizeof(float));
    
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v1[i] - v2[i];
    }
    return result;
}


float vector_sum(float *v, const size_t dim)
{
    float sum = 0.0f;

    for (size_t i = 0; i < dim; i++)
    {
        sum += v[i];
    }
    return sum;
}


float *vector_scalar_product(float *v, float s, const size_t dim)
{
    float *result = calloc(dim, sizeof(float));
    
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v[i] * s;
    }
    return result;
}


float *vector_elementwise_product(float *v1, float *v2, const size_t dim)
{
    float *result  = calloc(dim, sizeof(float));
    for (size_t i = 0; i < dim; i++)
    {
        result[i] = v1[i] * v2[i];
    }
    return result;
}


float vector_dot_product(float *v1, float *v2, const size_t dim)
{
    float result = 0.0f;
    for (size_t i = 0; i < dim; i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}


float **vector_outer_product(float *col, float *row, const size_t col_dim,
        const size_t row_dim)
{
    float **matrix = malloc(col_dim * sizeof(float *));

    for (size_t i = 0; i < col_dim; i++)
    {
        matrix[i] = calloc(row_dim, sizeof(float));
        for (size_t j = 0; j < row_dim; j++)
        {
            matrix[i][j] = col[i] * row[j];
        }
    }
    return matrix;
}


float *matrix_vector_dot_product(float **m, float *v, const size_t n_rows, 
        const size_t n_cols)
{
    float *result = calloc(n_rows, sizeof(float));

    for (size_t r = 0; r < n_rows; r++)
    {
        for (size_t c = 0; c < n_cols; c++)
        {
            result[r] += m[r][c] * v[c];
        }
    }
    return result;
}


float **matmul(float **m1, float **m2, const size_t n_rows_m1,
        const size_t n_cols_m1, const size_t n_cols_m2)
{
    float **result;
    result = malloc(n_rows_m1 * sizeof(float *));

    for (size_t i = 0; i < n_rows_m1; i++)
    {
        result[i] = calloc(n_cols_m2, sizeof(float));
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


float **matrix_transpose(float **m, const size_t n_rows, const size_t n_cols)
{
    float **transposed_matrix = malloc(n_cols * sizeof(float *));
    for (size_t i = 0; i < n_cols; i++)
    {
        transposed_matrix[i] = calloc(n_rows, sizeof(float));
        for (size_t j = 0; j < n_rows; j++)
        {
            transposed_matrix[i][j] = m[j][i];
        }
    }
    return transposed_matrix;
}

