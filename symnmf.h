#ifndef SYMNMF_H
#define SYMNMF_H

double** sym(double** x_matrix, int n, int d);
double** ddg(double** a_matrix, int n);
double** norm(double** a_matrix, double** d_matrix, int n);
double** symnmf(double** w_matrix, double** h_matrix, int n, int k);

#endif