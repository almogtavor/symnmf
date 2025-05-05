import sys

import numpy as np
from sklearn.metrics import silhouette_score

import symnmf
from kmeans import kmeans
from utils import init_h_matrix, load_valid_input

np.random.seed(1234)


# The function performs the symnmf algorithm on k centroids, and returns silhouette score
def calc_symnmf(k, x_matrix):
    n = len(x_matrix)
    # compute normalized similarity (W) - this is a nxn list-of-lists
    w_matrix = symnmf.norm(x_matrix)
    h_matrix = init_h_matrix(n, k, w_matrix)
    # initialize H in [0, 2*sqrt(m / k)]
    final_h_matrix = symnmf.symnmf(w_matrix, h_matrix, k)
    # applying 1.5 (cluster assignments) according to H (a list of lists, nxk)
    labels = [int(max(range(k), key=lambda j: final_h_matrix[i][j])) for i in range(n)]
    return silhouette_score(x_matrix, labels)


def run_kmeans_silhouette(k, x_matrix):
    dim = len(x_matrix[0])
    labels = kmeans(k, dim, x_matrix)
    return silhouette_score(x_matrix, labels)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError
        clusters_k, points = load_valid_input(sys.argv[1], sys.argv[2])
        if clusters_k <= 1:  # k should be larger than 1 (k > 1)
            raise ValueError
        nmf_score = calc_symnmf(clusters_k, points)
        kmeans_score = run_kmeans_silhouette(clusters_k, points)
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")
    except Exception as e:
        print(f"An Error Has Occurred")
        sys.exit(1)
