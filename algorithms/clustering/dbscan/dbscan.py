import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

'''
the code is based on
https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/,
'''


def points_in_circle(r, n=100):
    """Function for creating ring data points"""
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30, 30),
             math.sin(2*math.pi/n*x)*r+np.random.normal(-30, 30)) for x in range(1, n+1)]


def plot_the_data(df, title_text, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.scatter(df[0], df[1], s=15, **kwargs)
    plt.title(title_text, fontsize=20)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.show()


def k_means_cluster(df):
    k_means = KMeans(n_clusters=4, random_state=42)
    k_means.fit(df[[0, 1]])
    df['KMeans_labels'] = k_means.labels_


def hierarchical_clustering(df):
    model = AgglomerativeClustering(n_clusters=4, affinity='euclidean')
    model.fit(df[[0, 1]])
    df['HR_labels'] = model.labels_


def dbscan(df, **kwargs):
    dbscan = DBSCAN(**kwargs)
    dbscan.fit(df[[0, 1]])
    df['DBSCAN_labels'] = dbscan.labels_


def nearest_neighbors(df):
    """
    :param df:
    :return: distances between a data point and its nearest data point for all data points in the dataset.
    """
    neigh = NearestNeighbors(n_neighbors=2)
    neighbors = neigh.fit(df[[0, 1]])
    distances, indices = neighbors.kneighbors(df[[0, 1]])
    return distances


def plot_k_distance(distances):
    """
    plot the K-distance graph in order to optimize epsilon.
    :param distances: distance between a point and its nearest data point for all
    data points in the dataset using NearestNeighbors algorithm
    """
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(figsize=(20, 10))
    plt.plot(distances)
    plt.title('K-distance Graph', fontsize=20)
    plt.xlabel('Data Points sorted by distance', fontsize=14)
    plt.ylabel('Epsilon', fontsize=14)
    plt.show()


def main():
    np.random.seed(42)
    '''Creating ring data points'''
    df = pd.DataFrame(points_in_circle(500, 1000))
    df = df.append(points_in_circle(300, 700))
    df = df.append(points_in_circle(100, 300))
    '''Adding samples out of the rings ("noise")'''
    df = df.append([(np.random.randint(-600, 600), np.random.randint(-600, 600)) for i in range(300)])
    '''visualize'''
    plot_the_data(df, 'Dataset', color='grey')
    '''K-Means vs. Hierarchical vs. DBSCAN Clustering'''
    color_pallet = ['purple', 'red', 'blue', 'green']
    '''K-means fails to cluster the data points into four clusters.'''
    k_means_cluster(df)
    plot_the_data(df, 'K-Means Clustering', c=df['KMeans_labels'], cmap=matplotlib.colors.ListedColormap(color_pallet))

    '''The hierarchical clustering algorithm also fails to cluster the data points properly.'''
    hierarchical_clustering(df)
    plot_the_data(df, 'Hierarchical Clustering', c=df['HR_labels'], cmap=matplotlib.colors.ListedColormap(color_pallet))

    '''DBSCAN defaults: epsilon is 0.5, and min_samples or minPoints is 5.'''
    dbscan(df)
    plot_the_data(df, 'DBSCAN Clustering\n(before optimizing the epsilon parameter)', c=df['DBSCAN_labels'],
                  cmap=matplotlib.colors.ListedColormap(color_pallet))
    '''All the data points are treated as noise. It is because the value of epsilon is very small.'''

    '''optimize epsilon using the K-distance graph- in order to plot a K-distance Graph, we need the distance between
     a point and its nearest data point for all data points in the dataset. We use the NearestNeighbors algorithm for
     that purpose.'''
    distances = nearest_neighbors(df)
    '''The optimum value of epsilon is at the point of maximum curvature in the K-Distance Graph, 
    which is 30 in this case.'''
    plot_k_distance(distances)
    ''' The value of minPoints depends on domain knowledge'''
    dbscan(df, eps=30, min_samples=6)
    plot_the_data(df, 'DBSCAN Clustering', c=df['DBSCAN_labels'],
                  cmap=matplotlib.colors.ListedColormap(color_pallet))


if __name__ == '__main__':
    main()