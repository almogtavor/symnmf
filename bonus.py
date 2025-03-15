import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def plot_elbow(k_values, inertia, elbow_k):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Average Dispersion')
    plt.title('Elbow Method for Optimal k')
    

    # We're doing xytext=(elbow_k + 1, inertia[elbow_k-1] + 100)
    #   to place the text 1 unit to the right of the elbow
    #   and 100 units above the elbow point.
    plt.annotate(f'Elbow Point', xy=(elbow_k, inertia[elbow_k-1]), 
                 xytext=(elbow_k + 1, inertia[elbow_k-1] + 100), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig('elbow.png')
    plt.show()


def plot_elbow_method():
    iris = load_iris()

    inertia = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        kmeans.fit(iris.data)
        inertia.append(kmeans.inertia_)

    # The instructions where to "Run the sklearn k-means algorithm
    #   for the values of k ranging from k=1 till k=10 and plot the inertia for each value of k using the matplotlib module"
    # But also to "n output elbow.png in the program folder"
    # Therefore we didn't plotted for each value of k (since it would override the file), 
    #   but it can easily get implemented using an additional for loop.
    elbow_k = 3
    plot_elbow(k_values=k_values, inertia=inertia, elbow_k=elbow_k)

if __name__ == "__main__":
    plot_elbow_method()
