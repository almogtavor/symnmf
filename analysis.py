import math
import sys

import numpy as np
from sklearn.metrics import silhouette_score

import symnmf
from kmeans import kmeans

MAX_ITER = 300
EPSILON = 0.0001


def initialize_2d_array(row, col):
    arr = [[0.0] * col for _ in range(row)]
    return arr


# The function returns the dimension of the vectors from the input file
def get_vector_dim(input_file):
    dim = 1
    input_file.seek(0)
    for line in input_file:
        if dim == 1:
            dim += line.count(',')
            break
    input_file.seek(0)
    return dim


# The function returns the number of points from the input file
def get_row_count(input_file):
    row_counter = 0
    input_file.seek(0)
    for _ in input_file:
        row_counter += 1
    input_file.seek(0)
    return row_counter


def create_data_points(file_name):
    points = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            points.append(line.strip().split(','))
    if not points:
        print("An Error Has Occurred 543543254325325")
        sys.exit(1)
    return [[float(value) for value in row] for row in points]



# The function converts a list of nk elements to an array of arrays of size nk
def list_to_mat(lst, n, k):
    return [lst[i * k:(i + 1) * k] for i in range(n)]


# The function performs the symnmf algorithm on k centroids, and returns silhouette score
def calc_symnmf(k, file_name):
    points = create_data_points(file_name)
    np.random.seed(1234)
    W = symnmf.norm(points)
    n = int(math.sqrt(len(W)))
    # w_sum = sum(W)
    w_sum = sum(sum(row) for row in W)
    W = list_to_mat(W, n, n)
    upper_bound = 2.0 * math.sqrt((w_sum / (n * n)) / k)
    H = np.random.uniform(low=0.0, high=upper_bound, size=n * k).reshape(n, k).tolist()
    H = symnmf.symnmf(W, H)
    H = list_to_mat(H, n, k)
    H = np.array(H)
    # Compute cluster for each point by the min distance from the first k points
    clusters = [np.argmax(row) for row in H]
    return silhouette_score(points, clusters)


def run_kmeans_silhouette(k, file_name):
    points = create_data_points(file_name)
    dim = len(points[0])
    centroids = kmeans(k, MAX_ITER, dim, points)
    # Assign points to closest centroid
    labels = []
    for point in points:
        min_dist = float('inf')
        closest = -1
        for i, centroid in enumerate(centroids):
            dist = math.sqrt(sum((point[j] - centroid[j]) ** 2 for j in range(dim)))
            if dist < min_dist:
                min_dist = dist
                closest = i
        labels.append(closest)
    return silhouette_score(points, labels)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("An Error Has Occurred 543542543254")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        if k <= 1: # k should be larger than 1 (k > 1)
            raise ValueError
    except ValueError:
        print("An Error Has Occurred 54325425")
        sys.exit(1)

    input_file_name = sys.argv[2]

    try:
        points = create_data_points(input_file_name)
        if k >= len(points):
            print("An Error Has Occurred 23232")
            sys.exit(1)

        nmf_score = calc_symnmf(k, input_file_name)
        kmeans_score = run_kmeans_silhouette(k, input_file_name)

    except Exception as e:
        print(f"An Error Has Occurred 9892, {e.__dict__}")
        sys.exit(1)

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")
