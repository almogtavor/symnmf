#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "symnmf.h"

#define DEF_EPSILON 0.001
#define DEF_ITERATION 300

/* Utility function: Convert Python list to C array */
double** pylist_to_carray(PyObject* py_list, int rows, int cols) {
    double** c_array = (double**) malloc(rows * sizeof(double*));
    int i, j;

    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(py_list, i);
        c_array[i] = (double*) malloc(cols * sizeof(double));

        for (j = 0; j < cols; j++) {
            c_array[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    return c_array;
}

/* Utility function: Convert C array to Python list */
PyObject* carray_to_pylist(double** c_array, int rows, int cols) {
    PyObject* py_list = PyList_New(rows);
    int i, j;

    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(c_array[i][j]));
        }
        PyList_SetItem(py_list, i, row);
    }

    return py_list;
}

/* Utility function: Free allocated C array */
void free_carray(double** c_array, int rows) {
    int i;
    for (i = 0; i < rows; i++) {
        free(c_array[i]);
    }
    free(c_array);
}

/* Sym Wrapper */
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    PyObject* x_matrix_list;
    int n, d;

    if (!PyArg_ParseTuple(args, "Oii", &x_matrix_list, &n, &d)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for sym");
        return NULL;
    }

    double** x_matrix = pylist_to_carray(x_matrix_list, n, d);
    double** result = sym(x_matrix, n, d);

    PyObject* py_result = carray_to_pylist(result, n, n);

    free_carray(x_matrix, n);
    free_carray(result, n);

    return py_result;
}

/* DDG Wrapper */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    PyObject* a_matrix_list;
    int n;

    if (!PyArg_ParseTuple(args, "Oi", &a_matrix_list, &n)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for ddg");
        return NULL;
    }

    double** a_matrix = pylist_to_carray(a_matrix_list, n, n);
    double** result = ddg(a_matrix, n);

    PyObject* py_result = carray_to_pylist(result, n, n);

    free_carray(a_matrix, n);
    free_carray(result, n);

    return py_result;
}

/* Norm Wrapper */
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    PyObject* a_matrix_list;
    PyObject* d_matrix_list;
    int n;

    if (!PyArg_ParseTuple(args, "OOi", &a_matrix_list, &d_matrix_list, &n)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for norm");
        return NULL;
    }

    double** a_matrix = pylist_to_carray(a_matrix_list, n, n);
    double** d_matrix = pylist_to_carray(d_matrix_list, n, n);

    double** result = norm(a_matrix, d_matrix, n);

    PyObject* py_result = carray_to_pylist(result, n, n);

    free_carray(a_matrix, n);
    free_carray(d_matrix, n);
    free_carray(result, n);

    return py_result;
}

/* SymNMF Wrapper */
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    PyObject *w_matrix_list, *h_matrix_list;
    int n, k;

    if (!PyArg_ParseTuple(args, "OOii", &w_matrix_list, &h_matrix_list, &n, &k)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for symnmf");
        return NULL;
    }

    double** w_matrix = pylist_to_carray(w_matrix_list, n, n);
    double** h_matrix = pylist_to_carray(h_matrix_list, n, k);

    double** result = symnmf(w_matrix, h_matrix, n, k);

    PyObject* py_result = carray_to_pylist(result, n, k);

    free_carray(w_matrix, n);
    free_carray(h_matrix, n);
    free_carray(result, n);

    return py_result;
}

/* Module method table */
static PyMethodDef SymNMFMethods[] = {
    {"sym", sym_wrapper, METH_VARARGS, "Compute the Similarity Matrix"},
    {"ddg", ddg_wrapper, METH_VARARGS, "Compute the Diagonal Degree Matrix"},
    {"norm", norm_wrapper, METH_VARARGS, "Compute the Normalized Similarity Matrix"},
    {"symnmf", symnmf_wrapper, METH_VARARGS, "Run the SymNMF algorithm"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "Python interface for the SymNMF algorithm",
    -1,
    SymNMFMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
