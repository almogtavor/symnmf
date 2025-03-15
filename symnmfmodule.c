#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

#define DEF_EPSILON 0.001
#define DEF_ITERATION 300

static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    PyObject *init_centroids_list, *data_points_list;
    int k, iterations, vectors_num, cords_num;
    double epsilon;
    int *cluster_sizes;
    double **data_points, **initial_centroids, **final_centroids;

    /* Parse Python arguments - OO → Two Python objects (initial centroids & data points), i → Integer (number of clusters (k) & iterations), d → Double (epsilon) */
    if (!PyArg_ParseTuple(args, "OOiid", &init_centroids_list, &data_points_list, &k, &iterations, &epsilon)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments to fit function");
        return NULL;


    // Implement Python-C binding logic here
    Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef SymNMFMethods[] = {
    {"symnmf", symnmf_wrapper, METH_VARARGS, "Run the SymNMF algorithm"},
    {NULL, NULL, 0, NULL}  // Sentinel
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
