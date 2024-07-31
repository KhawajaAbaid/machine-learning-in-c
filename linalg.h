/*
 * Common Linear Algebra Ops
 *
 * All functions are pure.
 */
#pragma once

#include <stdlib.h>
#include <stddef.h>


double *vector_addition(double *v1, double *v2, const size_t dim);
double *vector_subtraction(double *v1, double *v2, const size_t dim);
double vector_sum(double *v, const size_t dim);
double *vector_scalar_product(double *v, double s, const size_t dim);
double *vector_scalar_addition(double *v, double s, const size_t dim);
double *vector_elementwise_product(double *v1, double *v2, const size_t dim);
double vector_dot_product(double *v1, double *v2, const size_t dim);
double **vector_outer_product(double *col, double *row, const size_t col_dim,
        const size_t row_dim);
double *matrix_vector_dot_product(double **m, double *v, const size_t n_rows, 
        const size_t n_cols);
double **matmul(double **m1, double **m2, const size_t n_rows_m1,
        const size_t n_cols_m1, const size_t n_cols_m2);
double **matrix_transpose(double **m, const size_t n_rows, const size_t n_cols);
double **matrix_scalar_product(double **m, double s, const size_t n_rows,
        const size_t n_cols);
double **matrix_scalar_addition(double **m, double s, const size_t n_rows,
        const size_t n_cols);
