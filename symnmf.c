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


double **handle_exception(void) {
    printf(ERROR_MSG);
    exit(1);
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

/* Matrix Transpose */
double **matrix_transpose(double **A, int rows, int cols) {
    int i, j;
    double **T = malloc(cols * sizeof(double *));
    for (i = 0; i < cols; i++) {
        T[i] = malloc(rows * sizeof(double));
        for (j = 0; j < rows; j++) {
            T[i][j] = A[j][i];
        }
    }
    return T;
}

/* Matrix Multiplication */
double **matrix_multiply(double **A, double **B, int A_rows, int A_cols, int B_cols) {
    int i, j, k;
    double **C = malloc(A_rows * sizeof(double *));
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

double calc_distance(double *point1, double *point2, int cords_num) {
    double sum = 0;
    int i;
    for (i = 0; i < cords_num; i++) {
        sum += (double) (point1[i] - point2[i]) * (double) (point1[i] - point2[i]);
    }
    /* We don't use sqrt here since in symnmf we calculate the squared Euclidean distance */
    return sum;
}

/* The Similarity Matrix */
double **sym(double **x_matrix, int n, int d) {
    double **a_matrix = malloc(n * sizeof(double *));
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

/* The Diagonal Degree Matrix Implementation */
double **ddg(double **a_matrix, int n) {
    double **d_matrix = malloc(n * sizeof(double *));
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

/* Normalized Similarity Matrix */
double **norm(double **a_matrix, double **d_matrix, int n) {
    double **w_matrix = malloc(n * sizeof(double *));
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
    ret = malloc(n * sizeof(double *));

    if (ret != NULL) {
        handle_exception();
    }
    for (i = 0; i < n; i++) {
        ret[i] = malloc(k * sizeof(double));
        if (ret[i] != NULL) {
            handle_exception();
        }
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

/* Display clustering results */
void print_clusters(int *clusters, int n) {
    int i;
    for (i = 0; i < n; i++) {
        printf("%d\n", clusters[i] + 1);
    }
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

/* Function to read data points from file */
double **read_data(const char *file_name, int *n, int *d) {
    FILE *file;
    double **data;
    char line[MAX_LINE];
    int row, col, i;
    char *token;

    file = fopen(file_name, "r");
    if (!file) {
        handle_exception();
    }

    data = NULL;
    row = 0;
    *d = 0;

    while (fgets(line, MAX_LINE, file)) {
        if (row == 0) {
            /* Count dimensions */
            token = strtok(line, ",");
            while (token) {
                (*d)++;
                token = strtok(NULL, ",");
            }
        }

        data = realloc(data, (row + 1) * sizeof(double *));
        if (!data) {
            handle_exception();
        }

        data[row] = malloc((*d) * sizeof(double));
        if (!data[row]) {
            handle_exception();
        }

        /* Re-read the first tokenized line using fgets again */
        fseek(file, 0, SEEK_SET); /* Reset file pointer to the start */
        for (i = 0; i <= row; i++) {
            fgets(line, MAX_LINE, file);
        }

        token = strtok(line, ",");
        for (col = 0; col < *d; col++) {
            if (token) {
                data[row][col] = atof(token);
                token = strtok(NULL, ",");
            }
        }

        row++;
    }

    *n = row;
    fclose(file);
    return data;
}

/* Helper to validate if the file contains only allowed characters */
int validate_file_format(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF) {
        if ((c < '0' || c > '9') &&
            c != '.' && c != ',' &&
            c != '\n' && c != '\r' &&
            c != '-') {
            return 0;
        }
    }
    return 1;
}

int count_dimensions(const char *line) {
    int count = 1; /* at least one coordinate */
    const char *p = line;
    while (*p) {
        if (*p == ',') {
            count++;
        }
        p++;
    }
    return count;
}


/* Validate file format */
int validate_file(FILE *file) {
    if (file == NULL || !validate_file_format(file)) {
        printf(ERROR_MSG);
        if (file != NULL) fclose(file);
        return 0;
    }
    return 1;
}

/* Count vectors and dimensions */
int verify_file_dimensions(FILE *file, int *n, int *d) {
    char line[MAX_LINE_LENGTH];\
    *n = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (strlen(line) > 1) {
            (*n)++;
        }
    }
    if (*n == 0) return 0;
    rewind(file);
    if (!fgets(line, MAX_LINE_LENGTH, file)) return 0;
    *d = count_dimensions(line);
    rewind(file);
    return 1;
}

/* Free partially allocated data matrix */
void free_partial_data(double **data, int filled) {
    int i;
    if (data == NULL) return;
    for (i = 0; i < filled; i++) {
        if (data[i] != NULL) {
            free(data[i]);
        }
    }
    free(data);
}

/* Load data from file */
double **load_data(FILE *file, int n, int d) {
    double **data = malloc(n * sizeof(double *));
    char line[MAX_LINE_LENGTH];
    int i, j;
    for (i = 0; i < n; i++) {
        char *token;
        if (!fgets(line, MAX_LINE_LENGTH, file)) {
            free_partial_data(data, i);
            return NULL;
        }

        data[i] = (double *) malloc(d * sizeof(double));
        if (data[i] == NULL) {
            free_partial_data(data, i);
            return NULL;
        }

        token = strtok(line, ",\n");
        for (j = 0; j < d; j++) {
            if (token == NULL) {
                free_partial_data(data, i + 1);
                return NULL;
            }
            data[i][j] = atof(token);
            token = strtok(NULL, ",\n");
        }
    }
    return data;
}

/* Execute goal and return result matrix (caller must free) */
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

/* Print a matrix */
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


int main(int argc, char *argv[]) {
    FILE *file;
    char *goal, *filename;
    int n = 0, d = 0;
    double **data = NULL, **result = NULL;

    if (argc != 3) {
        printf(ERROR_MSG);
        return 1;
    }

    goal = argv[1];
    filename = argv[2];
    file = fopen(filename, "r");
    if (!validate_file(file)) return 1;
    rewind(file);
    if (!verify_file_dimensions(file, &n, &d)) {
        fclose(file);
        printf(ERROR_MSG);
        return 1;
    }

    data = load_data(file, n, d);
    fclose(file);
    if (data == NULL) {
        printf(ERROR_MSG);
        return 1;
    }
    result = execute_goal(goal, data, n, d);
    if (result == NULL) {
        free_matrix(data, n);
        printf(ERROR_MSG);
        return 1;
    }

    print_matrix(result, n);
    free_matrix(data, n);
    free_matrix(result, n);
    return 0;
}
