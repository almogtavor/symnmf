#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define def_epsilon 0.001
#define def_iteration 200

double **initialize_centroids(double **points, int k, int cords_num) {
    double **centroids;
    int i, j;

    centroids = malloc(sizeof(double*) * k);
    for (i = 0; i < k; i++) {
        centroids[i] = malloc(sizeof(double) * cords_num);
        for (j = 0; j < cords_num; j++) {
            centroids[i][j] = points[i][j];
        }
    }
    return centroids;
}

/* Function to check if a string is a positive integer */
int is_positive_integer(const char *str) {
    int i;
    if (str == NULL || *str == '\0') return 0; /* Empty string or NULL is invalid */
    for (i = 0; str[i] != '\0'; i++) {
        if (str[i] < '0' || str[i] > '9') return 0; /* Non-digit character found */
    }
    return 1; /* All characters are digits */
}


double calc_distance(double *point1, double *point2, int cords_num) {
    double sum = 0;
    int i;
    for (i = 0; i < cords_num; i++) {
        sum += (double)(point1[i] - point2[i]) * (double)(point1[i] - point2[i]);
    }
    return sqrt(sum);
}

int argmin(double *point, double **centroids, int k, int cords_num) {
    double min_dist = calc_distance(point, centroids[0], cords_num);
    int closest_index = 0;
    int i;
    for (i = 1; i < k; i++) {
        double dist = calc_distance(point, centroids[i], cords_num);
        if (dist < min_dist) {
            min_dist = dist;
            closest_index = i;
        }
    }
    return closest_index;
}

/* A function to calculate the new centroid by averaging the points in the cluster */
double *calculate_new_centroid(double **cluster, int cluster_size, int cords_num) {
    double *new_centroid;
    int i, j;

    new_centroid = malloc(sizeof(double) * cords_num);
    for (i = 0; i < cords_num; i++) {
        new_centroid[i] = 0.0;
    }
    for (i = 0; i < cluster_size; i++) {
        for (j = 0; j < cords_num; j++) {
            new_centroid[j] += cluster[i][j];
        }
    }
    for (i = 0; i < cords_num; i++) {
        new_centroid[i] /= cluster_size; /* Average the coordinates */
    }
    return new_centroid;
}

double **kmeans(int k, int iterations, int cords_num, double **points, int vectors_num, double epsilon, double **initial_centroids, int *cluster_sizes) {
    double **centroids, **prev_centroids, ***clusters;
    int i, j, l, curr_i, converged;

    /* Use the provided initial centroids instead of initializing from points */
    centroids = malloc(sizeof(double*) * k);
    prev_centroids = malloc(sizeof(double*) * k);
    for (i = 0; i < k; i++) {
        centroids[i] = malloc(sizeof(double) * cords_num);
        prev_centroids[i] = malloc(sizeof(double) * cords_num);
        for (j = 0; j < cords_num; j++) {
            centroids[i][j] = initial_centroids[i][j];
            prev_centroids[i][j] = initial_centroids[i][j];
        }
    }
    
    clusters = malloc(sizeof(double**) * k); /* Pointer of the array of clusters */

    /* Allocate memory for clusters */
    for (i = 0; i < k; i++) {
        clusters[i] = malloc(sizeof(double *) * vectors_num); /* Allocate memory for each cluster */
        for (j = 0; j < vectors_num; j++) {
            clusters[i][j] = malloc(sizeof(double) * cords_num); /* Allocate for each point in cluster */
        }
        cluster_sizes[i] = 0;
    }

    curr_i = 0;
    converged = 0;
    while (curr_i < iterations && converged == 0) {
        for (i = 0; i < k; i++) {
            cluster_sizes[i] = 0;
            for (j = 0; j < vectors_num; j++) {
                for (l = 0; l < cords_num; l++) {
                    clusters[i][j][l] = 0.0; /* Reset cluster values to 0 */
                }
            }
        }
        for (i = 0; i < vectors_num; i++) {
            int cluster_index, size;
            cluster_index = argmin(points[i], centroids, k, cords_num);
            size = cluster_sizes[cluster_index];
            for (l = 0; l < cords_num; l++) {
                clusters[cluster_index][size][l] = points[i][l];
            }
            cluster_sizes[cluster_index]++;
        }

        for (i = 0; i < k; i++) {
            double *new_centroid = calculate_new_centroid(clusters[i], cluster_sizes[i], cords_num);
            for (j = 0; j < cords_num; j++) {
                prev_centroids[i][j] = centroids[i][j];
                centroids[i][j] = new_centroid[j];
            }
            free(new_centroid); /* Free the temp new centroid */
        }
        converged = 1;
        for (i = 0; i < k; i++) {
            if (calc_distance(centroids[i], prev_centroids[i], cords_num) >= epsilon) {
                converged = 0;
                break;
            }
        }
        curr_i++;
    }    
    for (i = 0; i < k; i++) {
        free(prev_centroids[i]);
        for (j = 0; j < vectors_num; j++) {
            free(clusters[i][j]);
        }
        free(clusters[i]);
    }
    free(prev_centroids);
    free(clusters);

    return centroids; /* Return the centroids instead of clusters */
}

/* Function to validate the input file for allowed characters */
int validate_input() {
    char c;
    while ((c = getchar()) != EOF) {
        if ((c < '0' || c > '9') && c != '.' && c != ',' && c != '\n' && c != '-') {
            /* If the character is not a digit, decimal point, comma, newline, or minus sign */
            return 0;
        }
    }
    return 1;
}
