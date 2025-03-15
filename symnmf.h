#ifndef SYMNMF_H
#define SYMNMF_H

double** sym(double** X, int n, int d);
double** ddg(double** A, int n);
double** norm(double** A, int n);
double** symnmf(double** W, double** H, int n, int k, double epsilon, int max_iter);


#endif //SYMNMF_H
