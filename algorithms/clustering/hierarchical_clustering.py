import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc

sys.path.append(os.getcwd())
from generic_fun.get_data import config_param_path_in

'''
•	This algorithm starts with all the data points assigned to a cluster of their own. 
•	Then two nearest clusters are merged into the same cluster.
•	In the end, this algorithm terminates when there is only a single cluster left.
https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
'''


def main(path_in, file_name):
    # get the data
    data = pd.read_csv(fr'{path_in}\{file_name}')
    # normalize
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    # visualize the raw data
    plt.figure(figsize=(10, 7))
    plt.scatter(data_scaled['Milk'], data_scaled['Grocery'])
    plt.show()
    # draw the dendrogram to decide the number of clusters
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.show()
    '''The vertical line with maximum distance is the blue line and hence we can decide two clusters'''
    # apply hierarchical clustering for 2 clusters
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data_scaled)
    # visualize the two clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)
    plt.show()
    '''The advantage of not having to pre-define the number of clusters gives it quite an edge over k-Means.'''


if __name__ == '__main__':
    file_name = r'Wholesale_customers_data.csv'
    path_in = config_param_path_in()
    main(path_in, file_name)