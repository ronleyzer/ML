import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


def generate_data():
    # Generate some data
    # X, y_true = make_blobs(n_samples=400, centers=4,
    #                        cluster_std=0.60, random_state=0)
    # X = X[:, ::-1]  # flip axes for better plotting
    X, y = datasets.load_iris(return_X_y=True)
    return X


def create_gmm(X, n_components, random_state):
    gmm = GMM(n_components=n_components, random_state=random_state).fit(X)
    labels = gmm.predict(X)
    return labels, gmm


def visualize_gmm(labels, X):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis')
    plt.show()


def visualize_data(X):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
    plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()


def plot_silhouette_scores(n_clusters, silhouette_scores):
    fig = plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'bo-', color='black')
    fig.tight_layout()
    plt.xlabel('N. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()


def silhouette_scores(X, n_clusters, random_state):
    silhouette_score = []
    for k in n_clusters:
        # Set the model and its parameters
        labels, gmm = create_gmm(X, n_components=k, random_state=random_state)
        # Calculate Silhoutte Score and append to a list
        silhouette_score.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    return silhouette_score


def pca(df, n_components=2):
    # Reducing the dimensions of the data
    pca = PCA(n_components=n_components)
    pca_df = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_df)
    pca_df.columns = ['P1', 'P2']
    return pca_df


def standardize_and_normalize(X):
    # Standardize data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)
    # Normalizing the Data
    normalized_df = normalize(scaled_df)
    # # Converting the numpy array into a pandas DataFrame
    # normalized_df = pd.DataFrame(normalized_df)
    return normalized_df


def main(random_state, selected_component):
    # Generate some data
    X = generate_data()
    X = pd.DataFrame(data=X)
    X = standardize_and_normalize(X)
    # PCA
    X = pca(X, n_components=2)
    # visualize raw data
    visualize_data(X)
    # decide how many components using silhouette score
    n_clusters = range(2, 9)
    silhouette_score = silhouette_scores(X, n_clusters, random_state)
    plot_silhouette_scores(n_clusters, silhouette_score)
    # Generalizing E–M: Gaussian Mixture Models
    labels, gmm = create_gmm(X, n_components=selected_component, random_state=random_state)
    # visualize after GMM
    visualize_gmm(labels, X)
    plot_gmm(gmm, X, label=True, ax=None)


if __name__ == '__main__':
    random_state = 42
    selected_component = 3
    main(random_state, selected_component)
