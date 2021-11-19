import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

''' this code is base on Androw Neg ML curs on Coursera 
https://www.coursera.org/learn/machine-learning/lecture/Mwrni/developing-and-evaluating-an-anomaly-detection-system
https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-anomaly-detection-1233d23dba95
'''


def get_the_data():
    path_in = r'P:\ML\data\anommaly_detection'
    mat = loadmat(fr"{path_in}\ex8data1.mat")
    X = mat["X"]
    Xval = mat["Xval"]
    yval = mat["yval"]
    return X, yval


def plot_the_data(X):
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.show()


def estimate_gaussian(X):
    """
     This function estimates the parameters of a Gaussian distribution using the data in X
    """
    m = X.shape[0]
    # compute the mean for each feature
    sum_ = np.sum(X, axis=0)
    mu = 1 / m * sum_
    # compute the variances for each feature
    var = 1 / m * np.sum((X - mu) ** 2, axis=0)
    return mu, var


def multivariate_gaussian(X, mu, sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.
    """
    k = len(mu)
    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma2) ** 0.5)) * np.exp(
        -0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    return p


def visualize_fit(X, mu, sigma2):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
    p2 = multivariate_gaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
    contour_level = 10 ** np.array([np.arange(-20, 0, 3, dtype=np.float)]).T
    # plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.show()


def select_threshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers
    """
    best_epi = 0
    best_F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    epi_range = np.arange(pval.min(), pval.max(), stepsize)
    for epi in epi_range:
        '''true positive. false positive, false negative'''
        '''the label y = 1 corresponds to an anomalous example, and y = 0 corresponds to a normal example'''
        predictions = (pval < epi)[:, np.newaxis]
        tp = np.sum(predictions[yval == 1] == 1)
        fp = np.sum(predictions[yval == 0] == 1)
        fn = np.sum(predictions[yval == 1] == 0)

        # compute precision, recall and F1
        if tp+fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)
        if tp+fn == 0:
            rec = 0
        else:
            rec = tp / (tp + fn)

        if prec+rec == 0:
            F1 = 0
        else:
            F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epi = epi

    return best_epi, best_F1


def plot_optimal_threshold(X, mu, sigma2, epsilon, p):
    plt.figure(figsize=(8, 6))
    # plot the data
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    # potting of contour
    # X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
    # p2 = multivariate_gaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
    # contour_level = 10 ** np.array([np.arange(-20, 0, 3, dtype=np.float)]).T
    # plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)
    # Circling of anomalies
    outliers = np.nonzero(p < epsilon)[0]
    plt.scatter(X[outliers, 0], X[outliers, 1], marker="o", facecolor="none", edgecolor="r", s=70)
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.show()


def main():
    X, yval = get_the_data()
    plot_the_data(X)
    '''split the data to train, cross-validation and test. save all anomalies to CV, test'''


    '''estimate parameters (mean and variance) for the Gaussian model'''
    mu, sigma2 = estimate_gaussian(X)
    '''Now that you have estimated the Gaussian parameters, you can investigate
       which examples have a very high probability given this distribution and which
       examples have a very low probability.'''
    # '''visualize the fit'''
    # visualize_fit(X, mu, sigma2)
    '''for every sample compute its product of probability-density-functions over all the features 
    The low probability examples are more likely to be the anomalies in our dataset.'''
    pval = multivariate_gaussian(X, mu, sigma2)
    '''select threshold - One way to determine which
        examples are anomalies is to select the threshold ε using the F1 score on a cross validation set.'''
    epsilon, F1 = select_threshold(yval, pval)
    print("Best epsilon found using cross-validation:", epsilon)
    print("Best F1 on Cross Validation Set:", F1)
    '''Visualizing the optimal threshold'''
    plot_optimal_threshold(X, mu, sigma2, epsilon, pval)


if __name__ == '__main__':
    main()