#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

#define ERROR_MSG "An Error Has Occurred\n"
#define MAX_LINE_LENGTH 10000

/* Print error and exit */
double **handle_exception(void) {
    printf(ERROR_MSG);
    exit(1);
}

/* Safe malloc with error check */
void *safe_malloc(size_t n) {
    void *p = malloc(n);
    if (!p) handle_exception();
    return p;
}

/* A util function to free a matrix */
void free_matrix(double **matrix, int rows) {
    int i;
    if (matrix == NULL) return;
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* Allocate matrix with safe malloc */
double **allocate_matrix(int rows, int cols) {
    int i;
    double **mat = safe_malloc(rows * sizeof(double *));
    for (i = 0; i < rows; i++) {
        mat[i] = safe_malloc(cols * sizeof(double));
    }
    return mat;
}

/* Matrix Transpose */
double **matrix_transpose(double **A, int rows, int cols) {
    int i, j;
    double **T = allocate_matrix(cols, rows);
    for (i = 0; i < cols; i++) {
        for (j = 0; j < rows; j++) {
            T[i][j] = A[j][i];
        }
    }
    return T;
}

/* Matrix Multiplication */
double **matrix_multiply(double **A, double **B, int A_rows, int A_cols, int B_cols) {
    int i, j, k;
    double **C = safe_malloc(A_rows * sizeof(double *));
    for (i = 0; i < A_rows; i++) {
        C[i] = calloc(B_cols, sizeof(double));
        for (j = 0; j < B_cols; j++) {
            for (k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

/* Calculate squared Euclidean distance */
double calc_distance(double *point1, double *point2, int cords_num) {
    double sum = 0;
    int i;
    for (i = 0; i < cords_num; i++) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    /* I don't use sqrt here since in symnmf we calculate the squared Euclidean distance */
    return sum;
}

/* Helper to validate if the file contains only allowed characters */
int validate_file_format(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF) {
        if ((c < '0' || c > '9') && c != '.' && c != ',' && c != '\n' && c != '\r' && c != '-') {
            return 0;
        }
    }
    return 1;
}

int validate_file(FILE *file) {
    if (file == NULL || !validate_file_format(file)) {
        printf(ERROR_MSG);
        if (file != NULL) fclose(file);
        return 0;
    }
    return 1;
}

/* Count how many values are in a line (i.e., dimensions) */
int count_dimensions(const char *line) {
    int count = 1;
    const char *p = line;
    while (*p) {
        if (*p == ',') count++;
        p++;
    }
    return count;
}

/* Check number of rows and dimensions in file */
int verify_file_dimensions(FILE *file, int *n, int *d) {
    char line[MAX_LINE_LENGTH];
    *n = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (strlen(line) > 1) (*n)++;
    }
    if (*n == 0) return 0;
    rewind(file);
    if (!fgets(line, MAX_LINE_LENGTH, file)) return 0;
    *d = count_dimensions(line);
    rewind(file);
    return 1;
}

/* Load matrix from file */
double **load_data(FILE *file, int n, int d) {
    char *token;
    double **data = allocate_matrix(n, d);
    char line[MAX_LINE_LENGTH];
    int i, j;
    for (i = 0; i < n; i++) {
        if (!fgets(line, MAX_LINE_LENGTH, file)) {
            free_matrix(data, i);
            return NULL;
        }
        token = strtok(line, ",\n");
        for (j = 0; j < d; j++) {
            if (token == NULL) {
                free_matrix(data, i + 1);
                return NULL;
            }
            data[i][j] = atof(token);
            token = strtok(NULL, ",\n");
        }
    }
    return data;
}

/* Print a matrix with 4 decimal places */
void print_matrix(double **matrix, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < n - 1) printf(",");
        }
        printf("\n");
    }
}
