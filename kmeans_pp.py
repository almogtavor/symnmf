import math
import sys
import pandas as pd
import numpy as np
import random
import mykmeanssp

np.random.seed(1234)

Vector = list
DEF_EPSILON = 0.001
DEF_ITERATION = 300

# Computes the Euclidean distance
def calc_distance(vector1: Vector, vector2: Vector):
    return math.sqrt(sum((vector1[i] - vector2[i]) ** 2 for i in range(len(vector1))))

def choose_centroids(all_points, centroids, k):
    if k == 1:
        return centroids

    for _ in range(1, k):
        min_distances = []
        for _, row in all_points.iterrows():
            point = row.tolist()
            distances = [calc_distance(point[1:], centroid.tolist()[1:]) for _, centroid in centroids.iterrows()]
            min_distances.append(min(distances))

        total_distances = sum(min_distances)
        # Choose the next centroid based on probabilities
        probabilities = [dist / total_distances for dist in min_distances]
        # Store them correctly in the DataFrame
        all_points = all_points.assign(min_distance=min_distances, probability=probabilities)

        chosen_index = np.random.choice(all_points.index, p=probabilities)
        new_centroid = all_points.drop(columns=['min_distance', 'probability']).loc[chosen_index]
        centroids = pd.concat([centroids, new_centroid.to_frame().T], ignore_index=True)

        all_points = all_points.drop(columns=['min_distance', 'probability'])

    return centroids

def main():
    # Parse command-line arguments (Usage example: python3 kmeans_pp.py 3 100 0.01 data/input_1.txt data/input_2.txt)
    if len(sys.argv) not in [5, 6]:
        print("Invalid number of arguments!")
        return

    try:
        k = sys.argv[1]
        if (type(k) is str and not k.isdigit()) or int(k) <= 1:
            raise ValueError("Invalid number of clusters!")
        max_iter = sys.argv[2] if len(sys.argv) == 6 else DEF_ITERATION
        if (type(max_iter) is str and not max_iter.isdigit()) or not (1 < int(max_iter) < 1000):
            raise ValueError("Invalid maximum iteration!")
        eps = float(sys.argv[3])
        if eps < 0:
            raise ValueError("Invalid epsilon!")
        file1_name = sys.argv[4]
        file2_name = sys.argv[5]
        k = int(k)
        max_iter = int(max_iter)
    except ValueError as e:
        print(e)
        return

    try:
        file1 = pd.read_csv(file1_name, header=None)
        file2 = pd.read_csv(file2_name, header=None)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    all_points = pd.merge(file1, file2, how="inner", on=0)
    all_points = all_points.sort_values(by=0)
    if len(all_points) <= k:
        print("Invalid number of clusters!")
        return
    try:
        # Initialize centroids using k-means++
        random_index = np.random.choice(all_points.index)
        first_centroid = all_points.loc[[random_index]].copy()

        centroids = choose_centroids(all_points, first_centroid, k)        
        # Call the C extension's fit function. We remove the key column before
        data_points = all_points.iloc[:, 1:].values.tolist()
        initial_centroids = centroids.iloc[:, 1:].values.tolist()
        final_centroids = mykmeanssp.fit(initial_centroids, data_points, k, max_iter, eps)
        # Output results
        initial_indices = centroids[0].astype(int).tolist()
        print(",".join(map(str, initial_indices)))

        for centroid in final_centroids:
            print(",".join(map(lambda x: f"{x:.4f}", centroid)))
    except Exception as e:
        print("An Error Has Occurred")
        return
    
if __name__ == "__main__":
    main()
