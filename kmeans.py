import math
import sys
from typing import List

Vector = List[float]
DEF_EPSILON = 0.001
DEF_ITERATION = 200


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


def kmeans(k: int, iterations: int, cords_num: int, points: List[Vector], epsilon: float = DEF_EPSILON):
    centroids = initialize_centroids(points, k)
    curr_i = 0
    converged = False
    while curr_i < iterations and not converged:
        clusters: list[list[Vector]] = [[] for _ in range(k)]
        previous_centroids = [centroid[:] for centroid in centroids]
        for i in range(len(points)):
            cluster_index = argmin(points[i], centroids)
            clusters[cluster_index].append(points[i])

        for i in range(len(clusters)):
            centroids[i] = calculate_new_centroid(clusters[i], cords_num)
        curr_i += 1
        for i in range(len(centroids)):
            if calc_distance(centroids[i], previous_centroids[i]) < epsilon:
                converged = True
            else:
                converged = False
                break
    return centroids


def initialize_centroids(vectors: List[Vector], k: int) -> List[Vector]:
    if k > len(vectors):
        raise ValueError('k must be less than/ equal to the number of vectors')
    return vectors[:k]


def main():
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print("An Error Has Occurred")
        sys.exit(1)

    all_dots, cords_num, k, iterations, vectors_num = load_dots()

    centroids = kmeans(k, iterations, cords_num, all_dots)
    for i, centroid in enumerate(centroids):
        for j in range(len(centroid)):
            if (j<len(centroid)-1):
                print(f"{centroid[j]:.4f}", end=",")
            else:
                print(f"{centroid[j]:.4f}")


def load_dots():
    if len(sys.argv) == 4:
        k = sys.argv[1]
        iterations = sys.argv[2]
        data_path = sys.argv[-1]
    elif len(sys.argv) == 3:
        k = sys.argv[1]
        iterations = DEF_ITERATION
        data_path = sys.argv[-1]
    else:
        raise ValueError("Invalid args provided")

    # Validate iterations (ensure it's an integer and greater than 0)
    try:
        iterations = int(iterations)
        if iterations <= 1 or iterations >= 1000:
            raise ValueError("Invalid maximum iteration!")
    except ValueError:
        print("Invalid maximum iteration!")
        sys.exit(1)

    try:
        k = int(k)
        if k <= 1 or k > iterations:
            raise ValueError("Number of clusters must be greater than 0.")
    except ValueError:
        print("Invalid number of clusters!")
        sys.exit(1)

    try:
        with open(data_path, "r") as file:
            lines = file.readlines()
            vectors_num = len(lines)
            if k > vectors_num:
                print(f"Invalid number of clusters!")
                sys.exit(1)
            cords_num = len(lines[0].strip().split(","))

            all_dots = [[0.0 for _ in range(cords_num)] for _ in range(vectors_num)]
        with open(data_path, "r") as file:
            for j, line in enumerate(file):
                line_arr = line.split(",")

                for i in range(len(line_arr)):
                    try:
                        all_dots[j][i] = float(line_arr[i])
                    except ValueError:
                        print("An Error Has Occurred")
                        sys.exit(1)
    except FileNotFoundError:
        print("An Error Has Occurred")
        sys.exit(1)
    return all_dots, cords_num, k, iterations, vectors_num


if __name__ == "__main__":
    main()