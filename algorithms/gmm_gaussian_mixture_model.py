# ***** Example of Gaussian Mixture Model *****
# Source: dokumen.pub_probability-for-machine-learning-discover-how-to-harness-uncertainty-with-python-v19nbsped.pdf
from matplotlib import pyplot
from numpy import hstack
from numpy.random import normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from matplotlib.patches import Ellipse
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
# model = GaussianMixture(n_components=2, init_params='random', random_state=random_state )
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
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import seaborn as sns


def plotPerColumnDistribution(df):
    colors = {0: 'powderblue', 1: 'lightsalmon', 2: 'darkcyan', 3: 'mediumorchid', 4: 'plum', 5: 'black',
              6: 'powderblue', 7: 'lightsalmon', 8: 'darkcyan', 9: 'mediumorchid', 10: 'plum', 11: 'black'}
    plt.figure(figsize=(15, 10))
    for i, x in enumerate(df.columns):
        label = f'''
        Mean: {round(df[x].mean(),2)} 
        Std: {round(df[x].std(),2)}
        '''
        # Min: {round(df[x].min(),2)}
        # Q1: {round(df[x].quantile(0.25))}
        # Q2:{round(df[x].quantile(0.5),2)}
        # Q3:{round(df[x].quantile(0.75),2)}
        # Max:{round(df[x].max(),2)}
        # Kurtosis: {round(kurtosis(df[x]),2)}
        # Simmetry: {round(skew(df[x]),2)}
        plt.subplot(4, 3, i+1)
        # plt.title(x, fontsize=10)
        sns.distplot(df[x], color=colors[i], label=label)
        plt.legend(fontsize=6)
        plt.tight_layout()
    plt.tight_layout()
    plt.show()


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
    # # Converting the numpy array into a pandas DataFrame
    # normalized_df = pd.DataFrame(normalized_df)
    return normalized_df


def pca(df, n_components=2):
    # Reducing the dimensions of the data
    pca = PCA(n_components=n_components)
    pca_df = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_df)
    pca_df.columns = ['P1', 'P2']
    return pca_df


def visualizing(pca_df, n_components, random_state):
    # Visualizing the clustering
    plt.scatter(pca_df['P1'], pca_df['P2'],
                c=GaussianMixture(n_components=n_components, random_state=random_state).fit_predict(pca_df), cmap=plt.cm.winter, alpha=0.6)
    plt.show()


def fit_gmm_model(df, n_components, random_state):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)


def select_number_of_clustering(df, n_clusters):
    sils = []
    sils_err = []
    iterations = 20
    for n in n_clusters:
        tmp_sil = []
        for iteration in range(iterations):
            train, test = train_test_split(df, test_size=0.5, random_state=iteration)
            gmm = GaussianMixture(n, n_init=1, random_state=iteration).fit(train)
            labels = gmm.predict(train)
            sil = metrics.silhouette_score(train, labels, metric='euclidean')
            tmp_sil.append(sil)
        val = np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
        err = np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    return sils, sils_err

# def select_number_of_clustering(df, n_clusters):
#     silhouette_scores = []
#     for k in n_clusters:
#         # Set the model and its parameters
#         model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans', random_state=random_state)
#         # Fit the model
#         labels = model.fit_predict(df)
#         # Calculate Silhoutte Score and append to a list
#         silhouette_scores.append(metrics.silhouette_score(df, labels, metric='euclidean'))
#     return silhouette_scores


def select_number_of_clustering_using_silhouette_scores(n_clusters, sils, sils_err):
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")
    plt.show()


# def select_number_of_clustering_using_silhouette_scores(n_clusters, silhouette_scores):
#     fig = plt.figure()
#     plt.plot(n_clusters, silhouette_scores, 'bo-', color='black')
#     fig.tight_layout()
#     plt.xlabel('N. of clusters')
#     plt.ylabel('Silhouette Score')
#     plt.title('Identify the number of clusters using Silhouette Score')
#     plt.show()


def create_gmm_model(train, selected_cluster, random_state):
    # Set the model and its parameters - 4 clusters
    model = GaussianMixture(n_components=selected_cluster,  # this is the number of clusters
                            covariance_type='full',  # {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
                            max_iter=100,  # the number of EM iterations to perform. default=100
                            n_init=1,  # the number of initializations to perform. default = 1
                            init_params='kmeans',
                            # the method used to initialize the weights, the means and the precisions.
                            # {'random' or default='k-means'}
                            verbose=0,  # default 0, {0,1,2}
                            random_state=random_state  # for reproducibility
                            )

    # Fit the model and predict labels
    cluster = model.fit(train)
    labels = model.predict(train)
    return cluster, labels, model


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


def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
                    + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


def plot_gmm_js_performance(n_clusters, results, res_sigs):
    plt.errorbar(n_clusters, results, yerr=res_sigs)
    plt.title("Distance between Train and Test GMMs", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Distance")
    plt.show()


def performance_gmm(n_clusters, data, test_size, random_state):
    iterations = 20
    results = []
    res_sigs = []
    for n in n_clusters:
        dist = []

        for iteration in range(iterations):
            train, test = train_test_split(data, test_size=test_size, random_state=iteration)
            gmm_train = GaussianMixture(n, n_init=1, random_state=random_state).fit(train)
            gmm_test = GaussianMixture(n, n_init=1, random_state=random_state).fit(test)
            dist.append(gmm_js(gmm_train, gmm_test))
        selec = SelBest(np.array(dist), int(iterations / 5))
        result = np.mean(selec)
        res_sig = np.std(selec)
        results.append(result)
        res_sigs.append(res_sig)
    plot_gmm_js_performance(n_clusters, results, res_sigs)


def main(path_in, file_name, selected_cluster, random_state):
    # get and clean the data
    df = read_csv(path_in, file_name)
    df = clean_data(df)
    # plotPerColumnDistribution(df)
    df = standardize_and_normalize(df)

    # split the data for unsupervised data
    train, test = train_test_split(df, test_size=0.5, random_state=random_state)

    # select number of clustering
    n_clusters = np.arange(2, 8)
    sils, sils_err = select_number_of_clustering(df, n_clusters)

    # visualize silhouette_scores - the higher the Silhouette score, the better defined your clusters are.
    # select_number_of_clustering_using_silhouette_scores(n_clusters, silhouette_scores)
    select_number_of_clustering_using_silhouette_scores(n_clusters, sils, sils_err)

    # after I choose the number of clusters, let's visualize
    pca_df = pca(train, n_components=2)
    visualizing(pca_df, selected_cluster, random_state)

    # after I choose the number of clusters, let’s now build our GMM model
    cluster, labels, model = create_gmm_model(pca_df, selected_cluster, random_state)
    print_gmm_model_results(cluster, selected_cluster)
    plot_gmm(model, pca_df)

    # performance - how much the GMMs trained ant test are similar-
    '''Jensen-Shannon (JS) - The lower the JS-distance, the better the cluster'''
    performance_gmm(n_clusters, data=df, test_size=0.5, random_state=random_state)


if __name__ == '__main__':
    path_in = r'P:\ML\kaggle\clustering'
    file_name = r'td_risk'
    selected_cluster = 3
    random_state = 6
    main(path_in, file_name, selected_cluster, random_state)