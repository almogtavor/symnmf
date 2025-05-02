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

/* 1.4.1 Initialize H */
double **initialize_H(int n, int k, double avg) {
    double **h_matrix;
    int i, j;

    h_matrix = (double **) malloc(n * sizeof(double *));

    for (i = 0; i < n; i++) {
        h_matrix[i] = (double *) malloc(k * sizeof(double));
        for (j = 0; j < k; j++) {
            /* By ((double)rand() / RAND_MAX) I'm getting a random num at the interval [0,1] */
            h_matrix[i][j] = ((double) rand() / RAND_MAX) * 2 * sqrt(avg / k);
        }
    }
    return h_matrix;
}

/* 1.4.2 Update H */
double **update_H(double **w_matrix, double **h_matrix, int n, int k) {
    int i, j, l, m;
    double **new_h = (double **) malloc(n * sizeof(double *));
    for (i = 0; i < n; i++) {
        new_h[i] = (double *) malloc(k * sizeof(double));
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

/* Free a matrix */
void free_matrix(double **matrix, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* SymNMF Main Function */
double **symnmf(double **w_matrix, double **h_matrix, int n, int k) {
    double **new_h;
    int iter, i, j;

    for (iter = 0; iter < MAX_ITER; iter++) {
        new_h = update_H(w_matrix, h_matrix, n, k);

        if (has_converged(h_matrix, new_h, n, k)) {
            free_matrix(new_h, n);
            break;
        }

        /* Copy new H to H for the next iteration */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                h_matrix[i][j] = new_h[i][j];
            }
        }
        /* Free the old `new_h` after copying its content */
        free_matrix(new_h, n);
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
        printf(ERROR_MSG);
        exit(1);
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
            printf(ERROR_MSG);
            exit(1);
        }

        data[row] = malloc((*d) * sizeof(double));
        if (!data[row]) {
            printf(ERROR_MSG);
            exit(1);
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

int main(int argc, char *argv[]) {
    FILE *file;
    char line[MAX_LINE_LENGTH];
    double **data = NULL, **a_matrix = NULL, **d_matrix = NULL, **result = NULL;
    int i, j, n = 0, d = 0;
    char *filename, *goal;

    if (argc != 3) {
        printf(ERROR_MSG);
        return 1;
    }

    goal = argv[1];
    filename = argv[2];

    file = fopen(filename, "r");
    if (file == NULL) {
        printf(ERROR_MSG);
        return 1;
    }

    /* Validate file format */
    if (!validate_file_format(file)) {
        fclose(file);
        printf(ERROR_MSG);
        return 1;
    }

    rewind(file);

    /* Count lines (vectors) */
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (strlen(line) > 1) {
            n++;
        }
    }

    if (n == 0) {
        fclose(file);
        printf(ERROR_MSG);
        return 1;
    }

    rewind(file);
    fgets(line, MAX_LINE_LENGTH, file);
    d = count_dimensions(line);

    data = (double **) malloc(n * sizeof(double *));
    if (data == NULL) {
        fclose(file);
        printf(ERROR_MSG);
        return 1;
    }

    rewind(file);
    for (i = 0; i < n; i++) {
        char *token;
        if (!fgets(line, MAX_LINE_LENGTH, file)) {
            printf(ERROR_MSG);
            return 1;
        }

        data[i] = (double *) malloc(d * sizeof(double));
        if (data[i] == NULL) {
            printf(ERROR_MSG);
            return 1;
        }

        token = strtok(line, ",\n");
        for (j = 0; j < d; j++) {
            if (token == NULL) {
                printf(ERROR_MSG);
                return 1;
            }
            data[i][j] = atof(token);
            token = strtok(NULL, ",\n");
        }
    }

    fclose(file);

    if (strcmp(goal, "sym") == 0) {
        result = sym(data, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        a_matrix = sym(data, n, d);
        result = ddg(a_matrix, n);
        /* Free a_matrix since it's only needed here */
        free_matrix(a_matrix, n);
    } else if (strcmp(goal, "norm") == 0) {
        a_matrix = sym(data, n, d);
        d_matrix = ddg(a_matrix, n);
        result = norm(a_matrix, d_matrix, n);
        free_matrix(a_matrix, n);
        free_matrix(d_matrix, n);
    } else {
        printf(ERROR_MSG);
        free_matrix(data, n);
        return 1;
    }

    /* Print result matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.4f", result[i][j]);
            if (j < n - 1) printf(",");
        }
        printf("\n");
    }

    free_matrix(data, n);
    free_matrix(result, n);

    return 0;
}
