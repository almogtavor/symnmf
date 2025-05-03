#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPSILON 1e-4
#define MAX_ITER 300
#define BETA 0.5

#define MAX_LINE 1024
#define ERROR_MSG "An Error Has Occurred\n"
#define MAX_LINE_LENGTH 10000

double **sym(double **x_matrix, int n, int d);

double **ddg(double **a_matrix, int n);

double **norm(double **a_matrix, double **d_matrix, int n);

double **symnmf(double **w_matrix, double **h_matrix, int n, int k);

int validate_file(FILE *file);

double **execute_goal(const char *goal, double **data, int n, int d);

/* Print error and exit */
double **handle_exception(void) {
    printf(ERROR_MSG);
    exit(1);
}

/* Safe malloc with error check */
static void *safe_malloc(const size_t n) {
    void *p = malloc(n);
    if (!p) {
        handle_exception();
    }
    return p;
}

/* A util function to free a matrix */
void free_matrix(double **matrix, int n) {
    int i;
    if (matrix == NULL) return;
    for (i = 0; i < n; i++) {
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
    /* We don't use sqrt here since in symnmf we calculate the squared Euclidean distance */
    return sum;
}

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
            ret[i][j] = h_matrix[i][j] * (1 - BETA + (BETA * (WH[i][j] / H_H_T_H[i][j])));
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
