import math
import sys

import numpy as np
from sklearn.metrics import silhouette_score

import symnmf

MAX_ITER = 300
EPSILON = 0.0001

#The function creates a two-dimensional list by coordinates
def initialize_2d_array(row, col):
    arr = [[0.0] * col for _ in range(row)]
    return arr

#The function inserts all the points from the input file into a two-dimensional array
def set_points_in_table(input_file, k, dim, point_table, k_table):
    line = input_file.readline()
    point_num = 0

    while line:
        values = line.split(',')
        for vector_current_dim in range(dim):
            value = float(values[vector_current_dim].strip())
            if point_num < k:
                k_table[point_num][vector_current_dim] = value
            point_table[point_num][vector_current_dim] = value
        point_num += 1
        line = input_file.readline()
    input_file.seek(0)

#The function returns the dimension of the vectors from the input file
def get_vector_dim(input_file):
    dim = 1
    input_file.seek(0)
    for line in input_file:
        if dim == 1:
            dim += line.count(',')
            break
    input_file.seek(0)
    return dim

#The function returns the number of points from the input file
def get_row_count(input_file):
    row_counter = 0
    input_file.seek(0)
    for i in input_file:
        row_counter += 1
    input_file.seek(0)
    return row_counter

#The function returns the distance between 2 points
def get_distance(point, centroid, dim):
    distance = 0.0
    for i in range(dim):
        distance += (point[i] - centroid[i]) ** 2
    return math.sqrt(distance)

#The function adds a point to the post_iteration table
def add_point_to_post_iter_table(row, dim, centroid, post_iter_table, point_table):
    for i in range(dim):
        post_iter_table[centroid][i] += point_table[row][i]
    post_iter_table[centroid][dim] += 1

#The function assigns a centroid to each point by the minimum distance of the point from the centroids
def set_centroids_to_point(k, dim, row, point_table, k_table, post_iter_table):
    min_dist = get_distance(point_table[row], k_table[0], dim)
    point_centroid = 0

    for centroid in range(k):
        distance = get_distance(point_table[row], k_table[centroid], dim)
        if distance < min_dist:
            min_dist = distance
            point_centroid = centroid
            point_table[row][dim] = point_centroid

    add_point_to_post_iter_table(row, dim, point_centroid, post_iter_table, point_table)

#The function calculates the average of the points of each centroid
def calc_and_set_average(k, dim, post_iter_table):
    for row in range(k):
        if post_iter_table[row][dim] != 0:
            for col in range(dim):
                post_iter_table[row][col] /= post_iter_table[row][dim]

#The function calculates and returns the max distance from k points
def max_delta(k, dim, k_table, post_iter_table):
    max_val = 0.0
    for i in range(k):
        diff = get_distance(k_table[i], post_iter_table[i], dim)
        if diff > max_val:
            max_val = diff
    return max_val

#The function updates the newly calculated centroids
def update_centroid_to_k_table(k, dim, k_table, post_iter_table):
    for i in range(k):
        for j in range(dim):
            k_table[i][j] = post_iter_table[i][j]

#The function initializes the post_iter table
def reset_post_iter_table(k, dim, post_iter_table):
    for i in range(k):
        for j in range(dim + 1):
            post_iter_table[i][j] = 0

#The function initializes a two-dimensional array with the points from the file (intended for symnmf)
def create_data_points(filename):
    points = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            points.append(line.strip().split(','))
    points = [[float(value) for value in row] for row in points]
    return points

#The function performs the Kmeans algorithm on k centroids, and returns silhouette score
def kmeans(k,filename):
    input_file = open(filename, "r")
    dim = get_vector_dim(input_file)
    num_of_points = get_row_count(input_file)
    point_table = initialize_2d_array(num_of_points,dim+1)
    post_iter_table = initialize_2d_array(k, dim + 1)
    k_table = initialize_2d_array(k, dim)
    delta = 1 + EPSILON
    current_iter = 0
    set_points_in_table(input_file, k, dim, point_table, k_table)
    while current_iter <= MAX_ITER and delta > EPSILON:
        for i in range(num_of_points):
            set_centroids_to_point(k, dim, i, point_table, k_table, post_iter_table)
        calc_and_set_average(k, dim, post_iter_table)
        delta = max_delta(k, dim, k_table, post_iter_table)
        update_centroid_to_k_table(k, dim, k_table, post_iter_table)
        current_iter += 1
        if current_iter == MAX_ITER or delta < EPSILON:
            break
        reset_post_iter_table(k, dim, post_iter_table)
    centroid_assign = np.array(point_table)[:, -1]
    point_table = np.array(point_table)[:, :-1]
    return silhouette_score(point_table, centroid_assign)

#The function converts a list of nk elements to an array of arrays of size nk
def list_to_mat(lst,n,k):
    return [lst[i * k:(i + 1) * k] for i in range(n)]

#The function performs the symnmf algorithm on k centroids, and returns silhouette score
def calc_symnmf(k,filename):
    points = create_data_points(filename)
    np.random.seed(1234)
    W = symnmf.norm(points)
    n = int(math.sqrt(len(W)))
    w_sum = sum(W)
    W = list_to_mat(W,n,n)
    upper_bound = 2.0 * math.sqrt((w_sum / (n*n))/ k)
    H = np.random.uniform(low=0.0, high=upper_bound, size=n*k).reshape(n,k).tolist()
    H = symnmf.symnmf(W,H)
    H=list_to_mat(H,n,k)
    H = np.array(H)
    #Compute cluster for each point by the min distance from the first k points
    clustList = []
    for row in H:
        clustList.append(np.argmax(row))
    return sklearn.metrics.silhouette_score(points, clustList)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
    k = int(sys.argv[1])
    filename = sys.argv[2]
    kmeans_score = kmeans(k,filename)
    nmf_score = calc_symnmf(k,filename)
    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")