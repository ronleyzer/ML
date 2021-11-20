import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import scipy

''' this code is base on Androw Neg ML curs on Coursera 
https://www.coursera.org/learn/machine-learning/lecture/Mwrni/developing-and-evaluating-an-anomaly-detection-system
https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-anomaly-detection-1233d23dba95
'''


def get_the_data():
    path_in = r'P:\ML\data\anomaly_detection'
    mat = loadmat(fr"{path_in}\ex8data1.mat")
    # save the data in repo
    # scipy.io.savemat('anomaly_two_features_and_target.mat', mat)
    # f =  os.getcwd() + '/' + f"anomaly_two_features_and_target.mat"
    # mat = loadmat(os.getcwd() + '/' + f"anomaly_two_features_and_target.mat")
    X = mat["X"]
    yval = mat["yval"]
    return X, yval, mat


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


def plot_optimal_threshold(X, epsilon, p, title):
    plt.figure(figsize=(8, 6))
    # plot the data
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    outliers = np.nonzero(p < epsilon)[0]
    plt.scatter(X[outliers, 0], X[outliers, 1], marker="o", facecolor="none", edgecolor="r", s=70)
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.title(f"{title}")
    plt.show()


def split_train_cv_test_normal_and_anomaly(mat):
    '''
    this function take ndarray of the data, split to normal and non-normal and split the data so
    that train gets 60% of the normal samples, cv and test each gets 20% of normals, and 50% of anomaly samples
    :param mat: dict data with target
    :return:
    '''

    X = mat["Xval"]
    y = mat["yval"]

    normal = np.squeeze(mat["yval"] == 0)
    anomaly = np.squeeze(mat["yval"] == 1)

    X_normal = X[normal]
    X_anomaly = X[anomaly]
    y_normal = y[normal]
    y_anomaly = y[anomaly]

    np.random.seed(111)
    uniform_dis_normal = (np.random.uniform(size=len(X_normal)))[:, np.newaxis]
    uniform_dis_anomaly = (np.random.uniform(size=len(X_anomaly)))[:, np.newaxis]

    X_train = X_normal[np.squeeze(uniform_dis_normal <= 0.6)]
    X_cv = np.append(X_normal[np.squeeze((uniform_dis_normal > 0.6) & (uniform_dis_normal <= 0.8))],
                     X_anomaly[np.squeeze(uniform_dis_anomaly <= 0.5)], axis=0)
    X_test = np.append(X_normal[np.squeeze(uniform_dis_normal > 0.8)],
                       X_anomaly[np.squeeze(uniform_dis_anomaly > 0.5)], axis=0)

    y_train = y_normal[np.squeeze(uniform_dis_normal <= 0.6)]
    y_cv = np.append(y_normal[np.squeeze((uniform_dis_normal > 0.6) & (uniform_dis_normal <= 0.8))],
                     y_anomaly[np.squeeze(uniform_dis_anomaly <= 0.5)], axis=0)
    y_test = np.append(y_normal[np.squeeze(uniform_dis_normal > 0.8)],
                       y_anomaly[np.squeeze(uniform_dis_anomaly > 0.5)], axis=0)

    return X_train, X_cv, X_test, y_train, y_cv, y_test


def main():
    X, yval, mat = get_the_data()
    plot_the_data(X)
    '''split the data to train, cross-validation and test. save all anomalies to CV, test'''
    X_train, X_cv, X_test, y_train, y_cv, y_test = split_train_cv_test_normal_and_anomaly(mat)
    '''estimate parameters (mean and variance) for the Gaussian model'''
    mu, sigma2 = estimate_gaussian(X_train)
    '''Now that you have estimated the Gaussian parameters, you can investigate
       which examples have a very high probability given this distribution and which
       examples have a very low probability.'''
    '''for every sample compute its product of probability-density-functions over all the features 
    The low probability examples are more likely to be the anomalies in our dataset.'''
    pval = multivariate_gaussian(X_cv, mu, sigma2)
    '''select threshold - One way to determine which
        examples are anomalies is to select the threshold ε using the F1 score on a cross validation set.'''
    epsilon, F1 = select_threshold(y_cv, pval)
    print("Best epsilon found using cross-validation:", epsilon)
    print("Best F1 on Cross Validation Set:", F1)
    '''Visualizing the optimal threshold cv'''
    plot_optimal_threshold(X_cv, epsilon, pval, title="cv")
    '''Visualizing the optimal threshold test'''
    pval = multivariate_gaussian(X_test, mu, sigma2)
    plot_optimal_threshold(X_test, epsilon, pval, title="test")


if __name__ == '__main__':
    main()