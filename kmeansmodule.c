#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEF_EPSILON 0.001
#define DEF_ITERATION 300

/* Function declarations */
double **initialize_centroids(double **points, int k, int cords_num);
double calc_distance(double *point1, double *point2, int cords_num);
int argmin(double *point, double **centroids, int k, int cords_num);
double *calculate_new_centroid(double **cluster, int cluster_size, int cords_num);
double **kmeans(int k, int iterations, int cords_num, double **points, int vectors_num, double epsilon, double **initial_centroids, int *cluster_sizes);

static PyObject *fit(PyObject *self, PyObject *args) {
    PyObject *init_centroids_list, *data_points_list;
    int k, iterations, vectors_num, cords_num;
    double epsilon;
    int *cluster_sizes;
    double **data_points, **initial_centroids, **final_centroids;

    /* Parse Python arguments - OO → Two Python objects (initial centroids & data points), i → Integer (number of clusters (k) & iterations), d → Double (epsilon) */
    if (!PyArg_ParseTuple(args, "OOiid", &init_centroids_list, &data_points_list, &k, &iterations, &epsilon)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments to fit function");
        return NULL;
    }

    vectors_num = PyObject_Length(data_points_list);
    cords_num = PyObject_Length(PyList_GetItem(data_points_list, 0));

    data_points = (double **)malloc(vectors_num * sizeof(double *));
    initial_centroids = (double **)malloc(k * sizeof(double *));
    final_centroids = (double **)malloc(k * sizeof(double *));
    cluster_sizes = (int *)malloc(k * sizeof(int));
    int i, j;

    for (i = 0; i < vectors_num; i++) {
        PyObject *vector = PyList_GetItem(data_points_list, i);
        data_points[i] = (double *)malloc(cords_num * sizeof(double));
        for (j = 0; j < cords_num; j++) {
            data_points[i][j] = PyFloat_AsDouble(PyList_GetItem(vector, j));
        }
    }

    for (i = 0; i < k; i++) {
        PyObject *vector = PyList_GetItem(init_centroids_list, i);
        initial_centroids[i] = (double *)malloc(cords_num * sizeof(double));
        for (j = 0; j < cords_num; j++) {
            initial_centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(vector, j));
        }
    }

    final_centroids = kmeans(k, iterations, cords_num, data_points, vectors_num, epsilon, initial_centroids, cluster_sizes);

    PyObject *result = PyList_New(k);
    for (i = 0; i < k; i++) {
        PyObject *centroid = PyList_New(cords_num);
        for (j = 0; j < cords_num; j++) {
            PyList_SetItem(centroid, j, PyFloat_FromDouble(final_centroids[i][j]));
        }
        PyList_SetItem(result, i, centroid);
        free(initial_centroids[i]);
        free(final_centroids[i]);
    }
    free(initial_centroids);
    free(final_centroids);
    free(cluster_sizes);
    for (i = 0; i < vectors_num; i++) {
        free(data_points[i]);
    }
    free(data_points);

    return result;
}

/* Module method function table */
static PyMethodDef kmeansMethods[] = {
    {"fit", fit, METH_VARARGS, "Run the k-means algorithm"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",  /* Module name in python */
    "Python interface for the k-means algorithm",  /* Module docstring as required */
    -1,  /* Module state */
    kmeansMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&kmeansmodule);
}
