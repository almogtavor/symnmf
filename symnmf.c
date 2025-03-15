#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

#define EPSILON 1e-4
#define MAX_ITER 300
#define BETA 0.5

double calc_distance(double *point1, double *point2, int cords_num) {
    double sum = 0;
    int i;
    for (i = 0; i < cords_num; i++) {
        sum += (double)(point1[i] - point2[i]) * (double)(point1[i] - point2[i]);
    }
    /* We don't use sqrt here since in symnmf we calculate the squared Euclidean distance */
    return sum;
}

/* The Similarity Matrix */
double** sym(double** x_matrix, int n, int d) {
    double** a_matrix = malloc(n * sizeof(double*));
    int i, j;
    for (i = 0; i < n; i++) {
        /* Calling calloc so all entries would set to 0 when allocated */
        a_matrix[i] = calloc(n, sizeof(double));
        for (j = 0; j < n; j++) {
            if (i != j) {
                a_matrix[i][j] = exp(-calc_distance(x_matrix[i], x_matrix[j], d) / 2.0);
            } else {
                a_matrix[i][j] = 0.0;
            }
        }
    }
    return a_matrix;
}

/* The Diagonal Degree Matrix Implementation */
double** ddg(double** a_matrix, int n) {
    double** d_matrix = malloc(n * sizeof(double*));
    int i, j;
    for (i = 0; i < n; i++) {
        d_matrix[i] = calloc(n, sizeof(double));
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += a_matrix[i][j];
        }
        d_matrix[i][i] = sum;
    }
    return d_matrix;
}

/* Normalized Similarity Matrix */
double** norm(double** a_matrix, double** d_matrix, int n) {
    double** w_matrix = malloc(n * sizeof(double*));
    int i, j;
    for (i = 0; i < n; i++) {
        w_matrix[i] = calloc(n, sizeof(double));
        for (j = 0; j < n; j++) {
            if (d_matrix[i][i] != 0 && d_matrix[j][j] != 0) {
                /* Since y/x is equivalent to 1/x * y */
                w_matrix[i][j] = a_matrix[i][j] / sqrt(d_matrix[i][i] * d_matrix[j][j]);
            }
        }
    }
    return w_matrix;
}

/* 1.4.1 Initialize H */
double** initialize_H(int n, int k, double avg) {
    double** h_matrix;
    int i, j;

    h_matrix = (double**) malloc(n * sizeof(double*));

    for (i = 0; i < n; i++) {
        h_matrix[i] = (double*) malloc(k * sizeof(double));
        for (j = 0; j < k; j++) {
            /* By ((double)rand() / RAND_MAX) I'm getting a random num at the interval [0,1] */
            h_matrix[i][j] = ((double)rand() / RAND_MAX) * 2 * sqrt(avg / k);
        }
    }
    return h_matrix;
}

/* 1.4.2 Update H */
double** update_H(double** w_matrix, double** h_matrix, int n, int k) {
    int i, j, l, m;
    double** new_h = (double**) malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        new_h[i] = (double*) malloc(k * sizeof(double));
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double WH_ij = 0.0;
            double HHTH_ij = 0.0;

            for (l = 0; l < n; l++) {
                WH_ij += w_matrix[i][l] * h_matrix[l][j];
            }

            for (m = 0; m < k; m++) {
                for (l = 0; l < n; l++) {
                    /* For each i,j I compute the multiplication of the i th row of H,
                    *  with all columns of H^T, multiplied by the j th column of H. */
                    HHTH_ij += h_matrix[i][m] * h_matrix[l][m] * h_matrix[l][j];
                }
            }
            new_h[i][j] = h_matrix[i][j] * (1 - BETA + BETA * WH_ij / HHTH_ij);
        }
    }
    return new_h;
}

/* 1.4.3 Convergence Check Using Frobenius Norm */
int has_converged(double** h_matrix, double** new_h, int n, int k) {
    int i, j;
    double sum_squared_diff = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double diff = new_h[i][j] - h_matrix[i][j];
            sum_squared_diff += diff * diff;
        }
    }

    return sum_squared_diff < EPSILON;
}

/* SymNMF Main Function */
double** symnmf(double** w_matrix, double** h_matrix, int n, int k) {
    double** new_h;
    int iter, i, j;

    for (iter = 0; iter < MAX_ITER; iter++) {
        new_h = update_H(w_matrix, h_matrix, n, k);

        if (has_converged(h_matrix, new_h, n, k)) {
            break;
        }

        /* Copy new H to H for the next iteration */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                h_matrix[i][j] = new_h[i][j];
            }
        }
    }

    return h_matrix;
}
