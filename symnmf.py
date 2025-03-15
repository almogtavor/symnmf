import sys
import numpy as np
import symnmf

def main():
    try:
        np.random.seed(1234)

        if len(sys.argv) != 4:
            print("An Error Has Occurred")
            return

        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]

        if file_name[-4:] != ".txt":
            print("An Error Has Occurred")
            return
        x_matrix = np.loadtxt(file_name)

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
            w_matrix = symnmf.norm(x_matrix.tolist())
            result = symnmf.symnmf(w_matrix, h_matrix.tolist(), k)
        else:
            print("An Error Has Occurred")
            return

        # Print result with 4 decimal precision
        for row in result:
            print(",".join(f"{val:.4f}" for val in row))
    except:
        print("An Error Has Occurred")
        return

if __name__ == "__main__":
    main()
