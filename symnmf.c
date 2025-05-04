#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"
#include "utils.h"

#define EPSILON 1e-4
#define ZERO_DIVISION_PROTECTOR_EPSILON 1e-6
#define MAX_ITER 300
#define BETA 0.5

#define MAX_LINE 1024
#define ERROR_MSG "An Error Has Occurred\n"
#define MAX_LINE_LENGTH 10000


/* 1.1 The Similarity Matrix */
double **sym(double **x_matrix, int n, int d) {
    double **a_matrix = safe_malloc(n * sizeof(double *));
    int i, j;
    for (i = 0; i < n; i++) {
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

/* 1.2 The Diagonal Degree Matrix Implementation */
double **ddg(double **a_matrix, int n) {
    double **d_matrix = safe_malloc(n * sizeof(double *));
    int i, j;
    double sum;
    for (i = 0; i < n; i++) {
        d_matrix[i] = calloc(n, sizeof(double));
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += a_matrix[i][j];
        }
        d_matrix[i][i] = sum;
    }
    return d_matrix;
}

/* 1.3 Normalized Similarity Matrix */
double **norm(double **a_matrix, double **d_matrix, int n) {
    double **w_matrix = safe_malloc(n * sizeof(double *));
    int i, j;
    for (i = 0; i < n; i++) {
        w_matrix[i] = calloc(n, sizeof(double));
        for (j = 0; j < n; j++) {
            if (d_matrix[i][i] != 0 && d_matrix[j][j] != 0) {
                w_matrix[i][j] = a_matrix[i][j] / sqrt(d_matrix[i][i] * d_matrix[j][j]);
            }
        }
    }
    return w_matrix;
}

/* 1.4.2 Update H */
double **update_H(double **w_matrix, double **h_matrix, int n, int k) {
    int i;
    int j;
    double **WH;
    double **H_T;
    double **H_H_T;
    double **H_H_T_H;
    double **ret;
    WH = matrix_multiply(w_matrix, h_matrix, n, n, k);
    H_T = matrix_transpose(h_matrix, n, k);
    H_H_T = matrix_multiply(h_matrix, H_T, n, k, n);
    H_H_T_H = matrix_multiply(H_H_T, h_matrix, n, n, k);
    ret = allocate_matrix(n, k);
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (H_H_T_H[i][j] == 0.0) {
                H_H_T_H[i][j] += ZERO_DIVISION_PROTECTOR_EPSILON;
            }
            ret[i][j] = h_matrix[i][j] *
                        (1 - BETA + (BETA * (WH[i][j] / H_H_T_H[i][j])));
        }
    }

    free_matrix(WH, n);
    free_matrix(H_T, k);
    free_matrix(H_H_T, n);
    free_matrix(H_H_T_H, n);
    return ret;
}

/* 1.4.3 Convergence Check Using Frobenius Norm */
int has_converged(double **h_matrix, double **new_h, int n, int k) {
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
double **symnmf(double **w_matrix, double **h_matrix, int n, int k) {
    double **new_h;
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        new_h = update_H(w_matrix, h_matrix, n, k);
        if (has_converged(h_matrix, new_h, n, k)) {
            free_matrix(h_matrix, n);
            return new_h;
        }
        free_matrix(h_matrix, n);
        h_matrix = new_h;
    }
    return h_matrix;
}

/* Execute goal */
double **execute_goal(const char *goal, double **data, int n, int d) {
    double **result = NULL, **a_matrix = NULL, **d_matrix = NULL;
    if (strcmp(goal, "sym") == 0) {
        result = sym(data, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        a_matrix = sym(data, n, d);
        result = ddg(a_matrix, n);
        free_matrix(a_matrix, n);
    } else if (strcmp(goal, "norm") == 0) {
        a_matrix = sym(data, n, d);
        d_matrix = ddg(a_matrix, n);
        result = norm(a_matrix, d_matrix, n);
        free_matrix(a_matrix, n);
        free_matrix(d_matrix, n);
    }
    return result;
}

int main(int argc, char *argv[]) {
    FILE *file;
    char *goal, *filename;
    int n = 0, d = 0;
    double **data = NULL, **result = NULL;

    if (argc != 3) handle_exception();
    goal = argv[1];
    filename = argv[2];
    file = fopen(filename, "r");
    if (!validate_file(file)) return 1;

    rewind(file);
    if (!verify_file_dimensions(file, &n, &d)) {
        fclose(file);
        handle_exception();
    }

    data = load_data(file, n, d);
    fclose(file);
    if (data == NULL) handle_exception();

    result = execute_goal(goal, data, n, d);
    if (result == NULL) {
        free_matrix(data, n);
        handle_exception();
    }
    print_matrix(result, n);
    free_matrix(data, n);
    free_matrix(result, n);
    return 0;
}
