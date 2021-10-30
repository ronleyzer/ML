# ***** Example of Gaussian Mixture Model *****
# Source: dokumen.pub_probability-for-machine-learning-discover-how-to-harness-uncertainty-with-python-v19nbsped.pdf
from matplotlib import pyplot
from numpy import hstack
from numpy.random import normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# # generate a sample
# # mean distribution = 20, std = 5, 3000 points
# X1 = normal(loc=20, scale=5, size=3000)
# # mean distribution = 40, std = 5, 7000 points
# X2 = normal(loc=40, scale=5, size=7000)
# X = hstack((X1, X2))
# # plot the histogram
# pyplot.hist(X, bins=50, density=True)
# '''We can see that for many of the points in the middle of the two peaks that it is ambiguous as to
# which distribution they were drawn from.'''
# pyplot.show()
#
# # fit model
# '''If the number of processes was not known, a range of different numbers of components could
# be tested and the model with the best fit could be chosen, where models could be evaluated using
# scores such as Akaike or Bayesian Information Criterion (AIC or BIC).
# init_params='random' randomly guess the initial parameters'''
# model = GaussianMixture(n_components=2, init_params='random', random_state=1 )
# X = X.reshape(-1, 1)
# model.fit(X)
#
# '''Once the model is fit, we can access the learned parameters via arguments on the model,
# such as the means, covariances, mixing weights, and more. More usefully, we can use the model
# to estimate the latent parameters for existing and new data points.'''
#
# # predict latent values
# yhat = model.predict(X)
# # check latent value for first few points
# print(yhat[:100])
# # check latent value for last few points
# print(yhat[-100:])

# ***** challenge
from sklearn.preprocessing import StandardScaler, normalize

''' Predict next-day rain in Australia '''
import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation

from sklearn.mixture import GaussianMixture  # for GMM clustering
from sklearn import metrics  # for calculating Silhouette score

import matplotlib.pyplot as plt  # for data visualization


def read_csv(path_in, file_name):
    df = pd.read_csv(rf'{path_in}\{file_name}.csv', encoding='utf-8',
                     index_col=[0], parse_dates=[0], dayfirst=True)
    return df


def clean_data(df):
    # For other columns with missing values, fill them in with column mean
    df = df.fillna(df.mean())
    return df


def standardize_and_normalize(df):
    # Standardize data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    # Normalizing the Data
    normalized_df = normalize(scaled_df)
    # Converting the numpy array into a pandas DataFrame
    normalized_df = pd.DataFrame(normalized_df)
    return normalized_df


def pca(df, n_components=2):
    # Reducing the dimensions of the data
    pca = PCA(n_components=n_components)
    pca_df = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_df)
    pca_df.columns = ['P1', 'P2']
    return pca_df


def visualizing(pca_df):
    # Visualizing the clustering
    plt.scatter(pca_df['P1'], pca_df['P2'],
                c=GaussianMixture(n_components=3, random_state=1).fit_predict(pca_df), cmap=plt.cm.winter, alpha=0.6)
    plt.show()


def fit_gmm_model(df, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=1)
    gmm.fit(df)


def select_number_of_clustering(df, n_clusters):
    silhouette_scores = []
    for k in n_clusters:
        # Set the model and its parameters
        model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans', random_state=1)
        # Fit the model
        labels = model.fit_predict(df)
        # Calculate Silhoutte Score and append to a list
        silhouette_scores.append(metrics.silhouette_score(df, labels, metric='euclidean'))
    return silhouette_scores


def select_number_of_clustering_using_silhouette_scores(n_clusters, silhouette_scores):
    fig = plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'bo-', color='black')
    fig.tight_layout()
    plt.xlabel('N. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()


def create_gmm_model(train, selected_cluster):
    # Set the model and its parameters - 4 clusters
    model = GaussianMixture(n_components=selected_cluster,  # this is the number of clusters
                            covariance_type='full',  # {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
                            max_iter=100,  # the number of EM iterations to perform. default=100
                            n_init=1,  # the number of initializations to perform. default = 1
                            init_params='kmeans',
                            # the method used to initialize the weights, the means and the precisions.
                            # {'random' or default='k-means'}
                            verbose=0,  # default 0, {0,1,2}
                            random_state=1  # for reproducibility
                            )

    # Fit the model and predict labels
    cluster = model.fit(train)
    labels = model.predict(train)
    return cluster, labels


def print_gmm_model_results(cluster, selected_cluster):
    print(f'*************** {selected_cluster} Cluster Model ***************')
    print('Means:\n ', cluster.means_)
    # print('Converged: ', cluster.converged_)
    # print('No. of Iterations: ', cluster.n_iter_)
    # print('Weights: ', cluster.weights_)
    # print('Covariances: ', cluster.covariances_)
    # print('Precisions: ', cluster.precisions_)
    # print('Precisions Cholesky: ', cluster.precisions_cholesky_)
    # print('Lower Bound: ', cluster.lower_bound_)


def main(path_in, file_name, selected_cluster):
    # get and clean the data
    df = read_csv(path_in, file_name)
    df = clean_data(df)
    df = standardize_and_normalize(df)

    # split the data for unsupervised data
    train, test = train_test_split(df, test_size=0.5)

    # select number of clustering
    n_clusters = np.arange(2, 8)
    silhouette_scores = select_number_of_clustering(train, n_clusters)

    # visualize silhouette_scores - the higher the Silhouette score, the better defined your clusters are.
    select_number_of_clustering_using_silhouette_scores(n_clusters, silhouette_scores)

    # after I choose the number of clusters, let's visualize
    pca_df = pca(train, n_components=selected_cluster)
    visualizing(pca_df)

    # after I choose the number of clusters, let’s now build our GMM model
    cluster, labels = create_gmm_model(pca_df, selected_cluster)
    print_gmm_model_results(cluster, selected_cluster)




    print("DONE")


if __name__ == '__main__':
    path_in = r'P:\ML\kaggle\clustering'
    file_name = r'Credit_Card_Dataset_for_Clustering'
    selected_cluster = 2
    main(path_in, file_name, selected_cluster)