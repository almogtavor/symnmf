import math
import sys
from typing import List

Vector = List[float]
MAX_ITER = 300
EPSILON = 0.0001


def is_positive_integer(value):
    return isinstance(value, int) and value > 0


def calc_distance(vector1: Vector, vector2: Vector):
    vectors_sum = 0
    for i in range(len(vector1)):
        vectors_sum += (vector1[i] - vector2[i]) ** 2
    return math.sqrt(vectors_sum)


def argmin(point: Vector, centroids: List[Vector]):
    min_dist = sys.maxsize
    index = -1
    for i in range(len(centroids)):
        dist = calc_distance(point, centroids[i])
        if dist < min_dist:
            min_dist = dist
            index = i
    return index


def calculate_new_centroid(cluster: list[Vector], cords_num: int) -> Vector:
    new_cent: Vector = [0.0] * cords_num
    for i in range(len(cluster)):
        for j in range(cords_num):
            new_cent[j] += cluster[i][j]
    for i in range(len(new_cent)):
        new_cent[i] = new_cent[i] / len(cluster)
    return new_cent


def kmeans(k: int, cords_num: int, points: List[Vector], epsilon: float = EPSILON):
    centroids = initialize_centroids(points, k)
    curr_i = 0
    converged = False
    labels = [0] * len(points)

    while curr_i < MAX_ITER and not converged:
        clusters: list[list[Vector]] = [[] for _ in range(k)]
        previous_centroids = [centroid[:] for centroid in centroids]

        for i in range(len(points)):
            cluster_index = argmin(points[i], centroids)
            clusters[cluster_index].append(points[i])
            labels[i] = cluster_index  # Save assignment
        for i in range(len(clusters)):
            if clusters[i]:  # Avoid division by zero
                centroids[i] = calculate_new_centroid(clusters[i], cords_num)

        curr_i += 1
        for i in range(len(centroids)):
            if calc_distance(centroids[i], previous_centroids[i]) < epsilon:
                converged = True
            else:
                converged = False
                break

    return labels

def initialize_centroids(vectors: List[Vector], k: int) -> List[Vector]:
    if k > len(vectors):
        raise ValueError('k must be less than/ equal to the number of vectors')
    return vectors[:k]
