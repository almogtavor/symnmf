import sys

import numpy as np
from sklearn.metrics import silhouette_score

import symnmf
from kmeans import kmeans

np.random.seed(1234)

def create_data_points(file_name):
    points = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            points.append(line.strip().split(','))
    if not points or len(points) == 0:
        print("An Error Has Occurred")
        sys.exit(1)
    return [[float(value) for value in row] for row in points]


# The function performs the symnmf algorithm on k centroids, and returns silhouette score
def calc_symnmf(k, file_name):
    x_matrix = create_data_points(file_name)
    n = len(x_matrix)
    # compute normalized similarity (W) - this is a nxn list-of-lists
    w_matrix = symnmf.norm(x_matrix)
    # compute m = average of all W[i][j]
    m = np.mean(w_matrix)
    h_matrix = np.random.uniform(0, 2 * np.sqrt(m / k),
                                 size=(len(x_matrix), k)).tolist()
    # initialize H in [0, 2*sqrt(m / k)]
    final_h_matrix = symnmf.symnmf(w_matrix, h_matrix, k)
    # applying 1.5 (cluster assignments) according to H (a list of lists, nxk)
    labels = [int(max(range(k), key=lambda j: final_h_matrix[i][j])) for i in range(n)]
    return silhouette_score(x_matrix, labels)


def run_kmeans_silhouette(k, file_name):
    points = create_data_points(file_name)
    dim = len(points[0])
    labels = kmeans(k, dim, points)
    return silhouette_score(points, labels)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError
        if not float(sys.argv[1]).is_integer():
            raise ValueError
        k = int(float(sys.argv[1]))
        if k <= 1:  # k should be larger than 1 (k > 1)
            raise ValueError

        input_file_name = sys.argv[2]
        if k >= len(create_data_points(input_file_name)):
            print("An Error Has Occurred")
            sys.exit(1)

        nmf_score = calc_symnmf(k, input_file_name)
        kmeans_score = run_kmeans_silhouette(k, input_file_name)
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")
    except Exception as e:
        print(f"An Error Has Occurred")
        sys.exit(1)
