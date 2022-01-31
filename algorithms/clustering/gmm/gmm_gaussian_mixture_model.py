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


def get_the_data():
    # Iris data
    X, y = datasets.load_iris(return_X_y=True)
    return X


def create_gmm(X, n_components, random_state):
    gmm = GMM(n_components=n_components, random_state=random_state).fit(X)
    labels = gmm.predict(X)
    return labels, gmm


def visualize_gmm(labels, X, title):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis')
    plt.title(f'{title}')
    plt.show()


def visualize_data(X, relevant_title):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
    plt.title(f'{relevant_title}')
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


def plot_gmm(title, labels, gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    if label:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.title(title)
    plt.show()


def plot_silhouette_scores(n_clusters, silhouette_scores):
    fig = plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'bo-', color='black')
    plt.xlabel('N. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score\n(by number of clusters)')
    plt.tight_layout()
    fig.tight_layout()
    plt.show()


def silhouette_scores(X, n_clusters, random_state):
    silhouette_score = []
    for k in n_clusters:
        '''Set the model and its parameters'''
        labels, gmm = create_gmm(X, n_components=k, random_state=random_state)
        '''Calculate Silhoutte Score and append to a list'''
        silhouette_score.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    return silhouette_score


def pca(df, n_components):
    '''Reducing the dimensions of the data'''
    pca = PCA(n_components=n_components)
    pca_df = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_df)
    return pca_df


def standardize_and_normalize(X):
    '''Standardize data'''
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)
    '''Normalizing the Data'''
    normalized_df = normalize(scaled_df)
    return normalized_df


def add_target(X):
    X['SNP Compounded'] = (pd.read_csv(r'P:\ML\kaggle\clustering\td_risk_y.csv', index_col=[0],
                                       parse_dates=[0], dayfirst=True).loc['2002-01-01':'2019-01-01']).iloc[:, 0]
    X = X.loc['2016-01-01':]
    sns.scatterplot(data=X, x=X.index, y='SNP Compounded', hue='label')
    plt.show()


def main(random_state, selected_component):
    '''get the data'''
    mat = get_the_data()
    X = pd.DataFrame(data=mat)
    X = standardize_and_normalize(X)
    '''PCA'''
    X = pca(X, n_components=2)

    '''visualize raw data'''
    title = 'Iris Data After Dimensionality Reduction\n(PCA with 2 components)'
    visualize_data(X, title)
    '''choose the number of components using silhouette score'''
    n_clusters = range(2, 9)
    silhouette_score = silhouette_scores(X, n_clusters, random_state)
    plot_silhouette_scores(n_clusters, silhouette_score)
    '''generalizing E–M: Gaussian Mixture Models'''
    labels, gmm = create_gmm(X, n_components=selected_component, random_state=random_state)
    '''visualize after GMM'''
    title = 'Iris Data Clustered with GMM'
    visualize_gmm(labels, X, title)
    plot_gmm(title, labels, gmm, X, label=True, ax=None)


if __name__ == '__main__':
    random_state = 42
    selected_component = 3
    main(random_state, selected_component)


