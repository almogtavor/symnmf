#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

void *safe_malloc(size_t n);
void free_matrix(double **matrix, int rows);
double **allocate_matrix(int rows, int cols);
double **matrix_transpose(double **A, int rows, int cols);
double **matrix_multiply(double **A, double **B, int A_rows, int A_cols, int B_cols);
double calc_distance(double *point1, double *point2, int cords_num);

int validate_file_format(FILE *file);
int validate_file(FILE *file);
int count_dimensions(const char *line);
int verify_file_dimensions(FILE *file, int *n, int *d);
double **load_data(FILE *file, int n, int d);
void print_matrix(double **matrix, int n);
double **handle_exception(void);

#endif
