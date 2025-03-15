import sys
import numpy as np
import symnmf

def main():
    np.random.seed(1234)

    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        return

    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]

    try:
        x_matrix = np.loadtxt(file_name)
    except:
        print("An Error Has Occurred")
        return

    if goal == "sym":
        result = symnmf.sym(x_matrix.tolist())
    elif goal == "ddg":
        result = symnmf.ddg(x_matrix.tolist())
    elif goal == "norm":
        result = symnmf.norm(x_matrix.tolist())
    elif goal == "symnmf":
        # Initialize h_matrix (H) for symnmf
        m = np.mean(x_matrix)
        h_matrix = np.random.uniform(0, 2 * np.sqrt(m / k), size=(x_matrix.shape[0], k))
        result = symnmf.symnmf(x_matrix.tolist(), h_matrix.tolist(), k)
    else:
        print("An Error Has Occurred")
        return

    # Print result with 4 decimal precision
    for row in result:
        print(",".join(f"{val:.4f}" for val in row))

if __name__ == "__main__":
    main()
