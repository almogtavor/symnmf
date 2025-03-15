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
        X = np.loadtxt(file_name)
    except:
        print("An Error Has Occurred")
        return

    if goal == "sym":
        result = symnmf.sym(X.tolist())
    elif goal == "ddg":
        result = symnmf.ddg(X.tolist())
    elif goal == "norm":
        result = symnmf.norm(X.tolist())
    elif goal == "symnmf":
        # Initialize H for symnmf
        m = np.mean(X)
        H = np.random.uniform(0, 2 * np.sqrt(m / k), size=(X.shape[0], k))
        result = symnmf.symnmf(X.tolist(), H.tolist(), k)
    else:
        print("An Error Has Occurred")
        return

    # Print result with 4 decimal precision
    for row in result:
        print(",".join(f"{val:.4f}" for val in row))

if __name__ == "__main__":
    main()
