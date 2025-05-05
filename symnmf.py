import sys

import numpy as np

import symnmf
from utils import init_h_matrix, load_valid_input

np.random.seed(1234)


def execute_goal(clusters_k, goal, x_matrix):
    if goal == "sym":
        result = symnmf.sym(x_matrix)
    elif goal == "ddg":
        result = symnmf.ddg(x_matrix)
    elif goal == "norm":
        result = symnmf.norm(x_matrix)
    elif goal == "symnmf":
        if clusters_k <= 1:
            raise ValueError
        # Initialize H (1.4.1), calculate W and run symnmf
        w_matrix = symnmf.norm(x_matrix)
        h_matrix = init_h_matrix(len(x_matrix), clusters_k, w_matrix)
        result = symnmf.symnmf(w_matrix, h_matrix, clusters_k)
    else:
        raise ValueError
    return result


def main():
    try:
        if len(sys.argv) != 4:
            raise ValueError
        clusters_k, x_matrix = load_valid_input(sys.argv[1], sys.argv[3])
        goal = sys.argv[2]
        result = execute_goal(clusters_k, goal, x_matrix)
        for row in result:
            print(",".join(f"{val:.4f}" for val in row))
    except Exception as e:
        print("An Error Has Occurred")
        return


if __name__ == "__main__":
    main()
