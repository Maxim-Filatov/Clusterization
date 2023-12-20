import numpy as np
import matplotlib.pyplot as plt
import math as m
# Used to generate the data points
from sklearn.datasets import make_blobs


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLNE = '\033[4m'
    END = '\033[0m'


def plot_data(X):
    plt.figure(figsize=(7.5, 6))
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color='k')


# Initially assigns a random data points as centroids for the clusters
def random_centroid():
    # Creating k random indices and using the data point at that indices as centroid
    centroids = [(8, 4), (6, 0)]
    # centroids = [(1, 4), (6, 3)]
    return centroids


# Determines which date point goes to which cluster
def assign_cluster(X, ini_centroids, k):
    # To store the data point's corresponding cluster number
    cluster = []
    # For every point in X
    for i in range(len(X)):
        # To store the distance between the centroid and data point
        euc_dist = []
        # Running k loops
        for j in range(k):
            # Appending the distance into the array
            euc_dist.append(np.linalg.norm(np.subtract(X[i], ini_centroids[j])))
        # Returns the index where the value is minimum
        idx = np.argmin(euc_dist)
        # Appends the index to the cluster array
        cluster.append(idx)
    return np.asarray(cluster)


# Returns the updated centroid
def compute_centroid(X, clusters, k):
    # Stores the centroid values
    centroid = []
    for i in range(k):
        temp_arr = []
        for j in range(len(X)):
            # Checking one cluster at once and storing the respective cluster data points in the temp_arr
            if clusters[j] == i:
                temp_arr.append(X[j])
        # Taking mean among those points and storing it in the centroid array
        centroid.append(np.mean(temp_arr, axis=0))
    return np.asarray(centroid)


# Return the difference between the previous centroid and the newly computed centroid
def difference(prev, nxt):
    diff = 0
    for i in range(len(prev)):
        diff += np.linalg.norm(prev[i]-nxt[i])
    return diff


# Used to plot in each iteration
def show_clusters(X, clusters, centroids, ini_centroids, mark_centroid=True, show_ini_centroid=True, show_plots=True):
    # Assigning specific color to each cluster. Assuming 3 for now
    cols = {0: 'r', 1: 'b', 2: 'g', 3: 'coral', 4: 'c', 5: 'lime'}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    # Plots every cluster points
    for i in range(len(clusters)):
        ax.scatter(X[i][0], X[i][1], color=cols[clusters[i]])
    # Plots all the centroids
    for j in range(len(centroids)):
        ax.scatter(centroids[j][0], centroids[j][1], marker='*', color=cols[j])
        if show_ini_centroid:
            ax.scatter(ini_centroids[j][0],ini_centroids[j][1], marker="+", s=150, color=cols[j])
    # Used to mark the centroid by drawing a circle around it
    if mark_centroid:
        for i in range(len(centroids)):
            ax.add_artist(plt.Circle((centroids[i][0], centroids[i][1]), 0.4, linewidth=2, fill=False))
            if show_ini_centroid:
                ax.add_artist(plt.Circle((ini_centroids[i][0], ini_centroids[i][1]), 0.4, linewidth=2, color='y', fill=False))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("K-means Clustering")
    if show_plots:
        plt.show()


# Used to perform k means clustering
# if show type input is not given then it will show plot for each loop
def k_means(X, k, show_type='all', show_plots=True):
    c_prev = random_centroid()  # initially assign a random centroid
    cluster = assign_cluster(X, c_prev, k)  # to store the cluster number of the data point
    diff = 100  # assuming that the initial difference between the centroids is 100
    ini_centroid = c_prev  # storing the initial centroid values

    print(
        Color.BOLD + "NOTE:\n + and Yellow Circle -> Initial Centroid\n * andBlack Circle -> Final Centroid" + Color.END)

    # stops if the difference is less than 0.001
    if show_plots:
        print(Color.BOLD + "\n\nInitial Plot:\n" + Color.END)
        show_clusters(X, cluster, c_prev, ini_centroid, show_plots=show_plots)
    while diff > 0.0001:
        cluster = assign_cluster(X, c_prev, k)  # assigns the data point to respective clusters
        # plotting the initial graph
        if show_type == 'all' and show_plots:
            show_clusters(X, cluster, c_prev, ini_centroid, False, False, show_plots=show_plots)
            mark_centroid = False  # Not to mark the centroids for other plots
            show_ini_centroid = False  # Not to mark the initial centroid for all
        c_new = compute_centroid(X, cluster, k)  # to compute the new centroid point
        diff = difference(c_prev, c_new)  # to compute the difference between the centroids
        c_prev = c_new  # now new centroid becomes current centroid point

    # Final cluster centers
    if show_plots:
        print(Color.BOLD + "\nInitial Cluster Centers:\n" + Color.END)
        print(ini_centroid)
        print(Color.BOLD + "\nFinal Cluster Centers:\n" + Color.END)
        print(c_prev)
        # Plotting the final plot
        print(Color.BOLD + "\n\nFinal Plot:\n" + Color.END)
        show_clusters(X, cluster, c_prev, ini_centroid, mark_centroid=True, show_ini_centroid=True)
    return cluster, c_prev


k = 2
X = [(3, 2), (1, 7), (0, 4), (8, 1), (5, 9)]
original_clus = make_blobs(n_samples=5, centers=2, n_features=2, random_state=20)

# plot_data(X)

cluster, centroid = k_means(X, k, show_type='ini_fin')
