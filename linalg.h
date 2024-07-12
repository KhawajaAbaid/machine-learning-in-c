/*
 * Common Linear Algebra Ops
 *
 * All functions are pure.
 */
#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
#include <stddef.h>


float *vector_addition(float *v1, float *v2, const size_t dim);
float *vector_subtraction(float *v1, float *v2, const size_t dim);
float vector_sum(float *v, const size_t dim);
float *vector_scalar_product(float *v, float s, const size_t dim);
float *vector_elementwise_product(float *v1, float *v2, const size_t dim);
float vector_dot_product(float *v1, float *v2, const size_t dim);
float **vector_outer_product(float *col, float *row, const size_t col_dim,
        const size_t row_dim);
float *matrix_vector_dot_product(float **m, float *v, const size_t n_rows, 
        const size_t n_cols);
float **matmul(float **m1, float **m2, const size_t n_rows_m1,
        const size_t n_cols_m1, const size_t n_cols_m2);
float **matrix_transpose(float **m, const size_t n_rows, const size_t n_cols);


#endif
