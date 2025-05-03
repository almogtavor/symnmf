#ifndef SYMNMF_H
#define SYMNMF_H

/**
 * calculate the similarity matrix by the formula: aij=e^(-||xi-xj||^2/2)
 *
 * @param x_matrix is a pointer to the points table
 * @param d is the dimension of the points from input
 * @param n is the number of points from input
 * @return the similarity matrix for the input
 */
double **sym(double **x_matrix, int n, int d);

double **ddg(double **a_matrix, int n);

double **norm(double **a_matrix, double **d_matrix, int n);

double **symnmf(double **w_matrix, double **h_matrix, int n, int k);

#endif