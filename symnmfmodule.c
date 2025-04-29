#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "symnmf.h"

#define DEF_EPSILON 0.001
#define DEF_ITERATION 300

/* Convert Python list of lists to C array */
double** pylist_to_carray(PyObject* py_list, int rows, int cols) {
    int i, j;
    double** c_array = (double**) malloc(rows * sizeof(double*));

    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(py_list, i);
        c_array[i] = (double*) malloc(cols * sizeof(double));
        for (j = 0; j < cols; j++) {
            c_array[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return c_array;
}

/* Convert C array to Python list of lists */
PyObject* carray_to_pylist(double** c_array, int rows, int cols) {
    int i, j;
    PyObject* py_list = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(c_array[i][j]));
        }
        PyList_SetItem(py_list, i, row);
    }
    return py_list;
}

/* Free a C array */
void free_carray(double** c_array, int rows) {
    int i;
    for (i = 0; i < rows; i++) {
        free(c_array[i]);
    }
    free(c_array);
}

/* sym(X) */
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    PyObject* x_matrix_list;
    double** x_matrix;
    double** result;
    PyObject* py_result;
    int n, d;

    if (!PyArg_ParseTuple(args, "O", &x_matrix_list)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for sym");
        return NULL;
    }

    n = PyList_Size(x_matrix_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Empty matrix");
        return NULL;
    }

    d = PyList_Size(PyList_GetItem(x_matrix_list, 0));

    x_matrix = pylist_to_carray(x_matrix_list, n, d);
    result = sym(x_matrix, n, d);
    py_result = carray_to_pylist(result, n, n);

    free_carray(x_matrix, n);
    free_carray(result, n);
    return py_result;
}

/* ddg(X) */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    PyObject* x_matrix_list;
    double** a_matrix;
    double** x_matrix;
    double** result;
    PyObject* py_result;
    int n;
    int d;

    if (!PyArg_ParseTuple(args, "O", &x_matrix_list)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for ddg");
        return NULL;
    }

    n = PyList_Size(x_matrix_list);
    d = PyList_Size(PyList_GetItem(x_matrix_list, 0));

    x_matrix = pylist_to_carray(x_matrix_list, n, d);
    a_matrix = sym(x_matrix, n, d);
    result = ddg(a_matrix, n);
    py_result = carray_to_pylist(result, n, n);

    free_carray(x_matrix, n);
    free_carray(a_matrix, n);
    free_carray(result, n);
    return py_result;
}

/* norm(X) */
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    PyObject* x_matrix_list;
    double** x_matrix;
    double** a_matrix;
    double** d_matrix;
    double** result;
    PyObject* py_result;
    int n;
    int d;

    if (!PyArg_ParseTuple(args, "O", &x_matrix_list)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for norm");
        return NULL;
    }

    n = PyList_Size(x_matrix_list);
    d = PyList_Size(PyList_GetItem(x_matrix_list, 0));

    x_matrix = pylist_to_carray(x_matrix_list, n, d);
    a_matrix = sym(x_matrix, n, d);
    d_matrix = ddg(a_matrix, n);
    result = norm(a_matrix, d_matrix, n);
    py_result = carray_to_pylist(result, n, n);

    free_carray(x_matrix, n);
    free_carray(a_matrix, n);
    free_carray(d_matrix, n);
    free_carray(result, n);
    return py_result;
}

/* symnmf(W, H, k) */
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    PyObject *w_matrix_list, *h_matrix_list;
    double** w_matrix;
    double** h_matrix;
    double** result;
    PyObject* py_result;
    int n, k;

    if (!PyArg_ParseTuple(args, "OOi", &w_matrix_list, &h_matrix_list, &k)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments for symnmf");
        return NULL;
    }

    n = PyList_Size(w_matrix_list);
    w_matrix = pylist_to_carray(w_matrix_list, n, n);
    h_matrix = pylist_to_carray(h_matrix_list, n, k);

    result = symnmf(w_matrix, h_matrix, n, k);
    py_result = carray_to_pylist(result, n, k);

    free_carray(w_matrix, n);
    free_carray(h_matrix, n);
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
    SymNMFMethods,
    NULL  /* Prevents the 'missing initializer' error */
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
