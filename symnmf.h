#ifndef SYMNMF_H
#define SYMNMF_H

/**
 * Calculate the similarity matrix by the formula: a_ij = exp(-||x_i - x_j||^2 / 2).
 *
 * @param x_matrix A 2D array of data points.
 * @param n The number of points in the input.
 * @param d The dimensionality of each point.
 * @return A pointer to the similarity matrix (n x n).
 */
double **sym(double **x_matrix, int n, int d);

/**
 * Compute the diagonal degree matrix D from the similarity matrix A.
 *
 * @param a_matrix The similarity matrix (n x n).
 * @param n The number of points.
 * @return A pointer to the diagonal degree matrix (n x n).
 */
double **ddg(double **a_matrix, int n);

/**
 * Compute the normalized similarity matrix W from A and D.
 *
 * @param a_matrix The similarity matrix (n x n).
 * @param d_matrix The diagonal degree matrix (n x n).
 * @param n The number of points.
 * @return A pointer to the normalized similarity matrix W (n x n).
 */
double **norm(double **a_matrix, double **d_matrix, int n);

/**
 * Perform the Symmetric Nonnegative Matrix Factorization - SymNMF algorithm.
 *
 * @param w_matrix The normalized similarity matrix W (n x n).
 * @param h_matrix The initial matrix H (n x k), will be updated iteratively.
 * @param n The number of data points.
 * @param k The number of clusters (columns in H).
 * @return A pointer to the final matrix H after convergence (n x k).
 */
double **symnmf(double **w_matrix, double **h_matrix, int n, int k);

#endif
