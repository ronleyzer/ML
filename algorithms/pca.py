import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

from algorithms.gmm_gaussian_mixture_model import standardize_and_normalize


def main():
    
    pass


def get_the_data():
    X, y = datasets.load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    return X


def visualize_the_explained_variance(explained_variance):
    cum_explained_variance = np.cumsum(explained_variance * 100)
    a = pd.DataFrame(cum_explained_variance)
    plt.scatter(a.index, a)
    # plt.plot(cum_explained_variance)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.show()


def pca(df, n_components, random_stat):
    # Reducing the dimensions of the data
    pca = PCA(n_components=n_components, random_state=random_stat)
    pca.fit(df)
    explained_variance = pca.explained_variance_ratio_
    pca_fit = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_fit)
    return explained_variance, pca_df


if __name__ == '__main__':
    X = get_the_data()
    standardize_and_normalize(X)
    # select the number of components to the pca
    explained_variance, pca_df = pca(X, n_components=X.shape[1], random_stat=111)
    visualize_the_explained_variance(explained_variance)

    print("HY!")