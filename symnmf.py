import os
import sys
import numpy as np
import symnmf

np.random.seed(1234)


def execute_goal(clusters_k, goal, x_matrix):
    if goal == "sym":
        result = symnmf.sym(x_matrix.tolist())
    elif goal == "ddg":
        result = symnmf.ddg(x_matrix.tolist())
    elif goal == "norm":
        result = symnmf.norm(x_matrix.tolist())
    elif goal == "symnmf":
        if clusters_k <= 1:
            raise ValueError
        # Initialize H (1.4.1), calculate W and run symnmf
        w_matrix = symnmf.norm(x_matrix.tolist())
        m = np.mean(w_matrix)
        h_matrix = np.random.uniform(0, 2 * np.sqrt(m / clusters_k),
                                     size=(x_matrix.shape[0], clusters_k))
        result = symnmf.symnmf(w_matrix, h_matrix.tolist(), clusters_k)
    else:
        raise ValueError
    return result


def main():
    try:
        if len(sys.argv) != 4:
            raise ValueError

        clusters_k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]

        if file_name[-4:] != ".txt" or os.path.getsize(file_name) == 0:
            raise ValueError
        x_matrix = np.genfromtxt(file_name, delimiter=',')
        # File must contain only valid float values, and k<n
        if x_matrix.size == 0 or np.isnan(x_matrix).any() or clusters_k >= x_matrix.shape[0]:
            raise ValueError
        result = execute_goal(clusters_k, goal, x_matrix)

        for row in result:
            print(",".join(f"{val:.4f}" for val in row))
    except Exception as e:
        print("An Error Has Occurred")
        return


if __name__ == "__main__":
    main()
