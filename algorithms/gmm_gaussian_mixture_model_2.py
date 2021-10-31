import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
import pandas as pd


def generate_data():
    # Generate some data
    X, y_true = make_blobs(n_samples=400, centers=4,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
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


def main():
    # Generate some data
    X = generate_data()
    X = pd.DataFrame(data=X)
    # visualize raw data
    visualize_data(X)
    # Generalizing E–M: Gaussian Mixture Models
    labels, gmm = create_gmm(X, n_components=4, random_state=42)
    # visualize after GMM
    visualize_gmm(labels, X)
    plot_gmm(gmm, X, label=True, ax=None)


    print("FY!")

if __name__ == '__main__':
    main()
