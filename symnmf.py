import sys
import numpy as np
import symnmf

def main():
    try:
        np.random.seed(1234)

        if len(sys.argv) != 4:
            print("An Error Has Occurred")
            return

        clusters_k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]

        if file_name[-4:] != ".txt":
            print("An Error Has Occurred")
            return
        x_matrix = np.genfromtxt(file_name, delimiter=',')
        n = x_matrix.shape[0]
        d = x_matrix.shape[1]

        if goal == "sym":
            result = symnmf.sym(x_matrix.tolist())
        elif goal == "ddg":
            result = symnmf.ddg(x_matrix.tolist())
        elif goal == "norm":
            result = symnmf.norm(x_matrix.tolist())
        elif goal == "symnmf":
            # Initialize h_matrix (H), and get w_matrx
            w_matrix = symnmf.norm(x_matrix.tolist())
            m = np.mean(w_matrix)
            h_matrix = np.random.uniform(0, 2 * np.sqrt(m / clusters_k),
                                         size=(x_matrix.shape[0], clusters_k))
            result = symnmf.symnmf(w_matrix, h_matrix.tolist(), clusters_k)
        else:
            print("An Error Has Occurred")
            return

        # Print result with 4 decimal precision
        for row in result:
            print(",".join(f"{val:.4f}" for val in row))
    except Exception as e:
        print("An Error Has Occurred")
        return

if __name__ == "__main__":
    main()
