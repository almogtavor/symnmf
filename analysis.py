import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import symnmf

def analyze(file_name, k):
    X = np.loadtxt(file_name)

    # SymNMF
    m = np.mean(X)
    H = np.random.uniform(0, 2 * np.sqrt(m / k), size=(X.shape[0], k))
    H_result = np.array(symnmf.symnmf(X.tolist(), H.tolist(), k))
    nmf_labels = np.argmax(H_result, axis=1)
    nmf_score = silhouette_score(X, nmf_labels)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=1234).fit(X)
    kmeans_score = silhouette_score(X, kmeans.labels_)

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
    else:
        analyze(sys.argv[2], int(sys.argv[1]))
