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
# model = GaussianMixture(n_components=2, init_params='random')
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
                c=GaussianMixture(n_components=3).fit_predict(pca_df), cmap=plt.cm.winter, alpha=0.6)
    plt.show()


def fit_gmm_model(df, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(df)


def select_number_of_clustering(df, n_clusters):
    silhouette_scores = []
    for k in n_clusters:
        # Set the model and its parameters
        model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
        # Fit the model
        labels = model.fit_predict(df)
        # Calculate Silhoutte Score and append to a list
        silhouette_scores.append(metrics.silhouette_score(df, labels, metric='euclidean'))
    return silhouette_scores


def select_number_of_clustering_using_silhouette_scores(n_clusters, silhouette_scores):
    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(n_clusters, silhouette_scores, 'bo-', color='black')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()


def main(path_in, file_name):
    # get and clean the data
    df = read_csv(path_in, file_name)
    df = clean_data(df)
    df = standardize_and_normalize(df)

    # split the data for unsupervised data
    train, test = train_test_split(df, test_size=0.2)

    # select number of clustering
    n_clusters = np.arange(2, 8)
    silhouette_scores = select_number_of_clustering(df, n_clusters)

    # visualize silhouette_scores - the higher the Silhouette score, the better defined your clusters are.
    select_number_of_clustering_using_silhouette_scores(n_clusters, silhouette_scores)

    # visualize
    pca_df = pca(train, n_components=2)
    visualizing(pca_df)


    print("DONE")


if __name__ == '__main__':
    path_in = r'P:\ML\kaggle\clustering'
    file_name = r'Credit_Card_Dataset_for_Clustering'
    main(path_in, file_name)