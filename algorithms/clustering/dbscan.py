import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

'''
Motivation: K-Means and Hierarchical Clustering both fail in creating clusters of arbitrary shapes. 
They are not able to form clusters based on varying densities. That’s why we need DBSCAN clustering.
•	DBSCAN is not just able to cluster the data points correctly, but it also perfectly detects noise in the dataset.
•	DBSCAN is robust to outliers.
•	It does not require the number of clusters to be told beforehand.
'''


def points_in_circle(r, n=100):
    '''Function for creating data points in the form of a circle'''
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
    '''
    The distance variable contains an array of distances between a data point and its nearest data point
    for all data points in the dataset.
    :param df:
    :return: distances
    '''
    neigh = NearestNeighbors(n_neighbors=2)
    neighbors = neigh.fit(df[[0, 1]])
    distances, indices = neighbors.kneighbors(df[[0, 1]])
    return distances


def plot_k_distance(distances):
    '''
    optimize epsilon using the K-distance graph
    :param distances: distance between a point and its nearest data point for all
    data points in the dataset using NearestNeighbors algorithm
    :return: figure
    '''
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
    # Creating data points in the form of a circle
    df = pd.DataFrame(points_in_circle(500, 1000))
    df = df.append(points_in_circle(300, 700))
    df = df.append(points_in_circle(100, 300))
    # Adding noise to the dataset
    df = df.append([(np.random.randint(-600, 600), np.random.randint(-600, 600)) for i in range(300)])
    # visualize
    plot_the_data(df, 'Dataset', color='grey')
    '''K-Means vs. Hierarchical vs. DBSCAN Clustering'''
    color_pallet = ['purple', 'red', 'blue', 'green']
    # K-means fails to cluster the data points into four clusters. Also, it did not work well with noise.
    k_means_cluster(df)
    plot_the_data(df, 'K-Means Clustering', c=df['KMeans_labels'], cmap=matplotlib.colors.ListedColormap(color_pallet))

    # The hierarchical clustering algorithm also failed to cluster the data points properly.
    hierarchical_clustering(df)
    plot_the_data(df, 'Hierarchical Clustering', c=df['HR_labels'], cmap=matplotlib.colors.ListedColormap(color_pallet))

    # DBSCAN defaults: epsilon is 0.5, and min_samples or minPoints is 5.
    dbscan(df)
    plot_the_data(df, 'DBSCAN Clustering', c=df['DBSCAN_labels'], cmap=matplotlib.colors.ListedColormap(color_pallet))
    '''All the data points are treated as noise. It is because the value of epsilon is very small.
    Lets optimize epsilon and minPoints and then train our model again.'''
    # optimize epsilon using the K-distance graph
    '''For plotting a K-distance Graph, we need the distance between a point and its nearest data point for all 
    data points in the dataset using NearestNeighbors algorithm.
    The optimum value of epsilon is at the point of maximum curvature in the K-Distance Graph, which is 30 in this case.
    '''
    distances = nearest_neighbors(df)
    plot_k_distance(distances)
    ''' The value of minPoints depends on domain knowledge. '''
    dbscan(df, eps=30, min_samples=6)
    plot_the_data(df, 'DBSCAN Clustering', c=df['DBSCAN_labels'],
                  cmap=matplotlib.colors.ListedColormap(color_pallet))


if __name__ == '__main__':
    main()