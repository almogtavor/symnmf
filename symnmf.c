#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-4
#define MAX_ITER 300
#define BETA 0.5

#define MAX_LINE 1024
#define ERROR_MSG "An Error Has Occurred\n"

double** sym(double** x_matrix, int n, int d);
double** ddg(double** a_matrix, int n);
double** norm(double** a_matrix, double** d_matrix, int n);
double** symnmf(double** w_matrix, double** h_matrix, int n, int k);

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
                double bla = -calc_distance(x_matrix[i], x_matrix[j], d);
                double ex = exp(-calc_distance(x_matrix[i], x_matrix[j], d) / 2.0);
                printf("before exp: %f \n", bla);
                printf("exp: %f \n", ex);
                a_matrix[i][j] = exp(-calc_distance(x_matrix[i], x_matrix[j], d) / 2.0);
            } else {
                a_matrix[i][j] = 0.0;
            }
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            printf("%.4f", a_matrix[i][j]);
            if (j < d - 1) printf(",");
        }
        printf("\n");
    }
    return a_matrix;
}

/* The Diagonal Degree Matrix Implementation */
double** ddg(double** a_matrix, int n) {
    double** d_matrix = malloc(n * sizeof(double*));
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

/* 1.5 - Hard Clustering Solution */
int* derive_clustering(double** h_matrix, int n, int k) {
    int* clusters;
    int i, j;

    clusters = (int*) malloc(n * sizeof(int));
    for (i = 0; i < n; i++) {
        double max_value = h_matrix[i][0];
        int max_index = 0;
        for (j = 1; j < k; j++) {
            if (h_matrix[i][j] > max_value) {
                max_value = h_matrix[i][j];
                max_index = j;
            }
        }
        clusters[i] = max_index;
    }
    return clusters;
}

/* Display clustering results */
void print_clusters(int* clusters, int n) {
    int i;
    for (i = 0; i < n; i++) {
        printf("%d\n", clusters[i] + 1);
    }
}

/* Free a matrix */
void free_matrix(double** matrix, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* SymNMF Main Function */
double** symnmf(double** w_matrix, double** h_matrix, int n, int k) {
    double** new_h;
/*    int* clusters;*/
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

/*    clusters = derive_clustering(h_matrix, n, k); */

/*    printf("Clustering Results:\n"); */
/*    print_clusters(clusters, n); */

/*    free(clusters); */
    return h_matrix;
}






#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define MAX_LINE 1024
#define ERROR_MSG "An Error Has Occurred\n"

/* Function to read data points from file */
double** read_data(const char* file_name, int* n, int* d) {
    FILE* file;
    double** data;
    char line[MAX_LINE];
    int row, col, i;
    char* token;

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

        data = realloc(data, (row + 1) * sizeof(double*));
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
        fseek(file, 0, SEEK_SET);  /* Reset file pointer to the start */
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

/* Main Function for Handling Arguments and Goals */
int main(int argc, char* argv[]) {
    int i, j, n, d;
    double** data;
    double** result;
    double** a_matrix;
    double** d_matrix;

    if (argc != 3) {
        printf(ERROR_MSG);
        return 1;
    }

    data = read_data(argv[2], &n, &d);

    if (strcmp(argv[1], "sym") == 0) {
        result = sym(data, n, d);

    } else if (strcmp(argv[1], "ddg") == 0) {
        a_matrix = sym(data, n, d);
        result = ddg(a_matrix, n);

        /* Free a_matrix since it's only needed here */
        for (i = 0; i < n; i++) {
            free(a_matrix[i]);
        }
        free(a_matrix);

    } else if (strcmp(argv[1], "norm") == 0) {
        a_matrix = sym(data, n, d);
        d_matrix = ddg(a_matrix, n);
        result = norm(a_matrix, d_matrix, n);

        /* Free a_matrix and d_matrix since they're only needed here */
        for (i = 0; i < n; i++) {
            free(a_matrix[i]);
            free(d_matrix[i]);
        }
        free(a_matrix);
        free(d_matrix);

    } else {
        printf(ERROR_MSG);
        return 1;
    }

    /* Print results */
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            printf("%.4f", result[i][j]);
            if (j < d - 1) printf(",");
        }
        printf("\n");
    }

    /* Free allocated memory */
    for (i = 0; i < n; i++) {
        free(data[i]);
        free(result[i]);
    }
    free(data);
    free(result);

    return 0;
}
