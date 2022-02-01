import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.append(os.getcwd())
from generic_fun.get_data import config_param_path_in

''' the code is based on Andrew Ng's ML course on Coursera 
https://www.coursera.org/learn/machine-learning/lecture/Mwrni/developing-and-evaluating-an-anomaly-detection-system
https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-anomaly-detection-1233d23dba95
'''


def get_the_data(path_in, file_name):
    mat = loadmat(fr"{path_in}\{file_name}")
    X = mat["X"]
    yval = mat["yval"]
    return X, yval, mat


def plot_the_data(X, title):
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.title(f'{title}')
    plt.grid()
    plt.show()


def estimate_gaussian(X):
    """
     This function estimates the parameters of a Gaussian distribution using the data in X
    """
    m = X.shape[0]
    '''compute the mean for each feature'''
    mu = (1 / m) * np.sum(X, axis=0)
    '''compute the variances for each feature'''
    var = (1 / m) * np.sum((X - mu) ** 2, axis=0)
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


def calc_proportion(nominator, additional_denominator):
    """return the nominator's proportion in nominator + additional_denominator
    in order to calculate precision and recall"""
    if nominator + additional_denominator == 0:
        proportion = 0
    else:
        proportion = nominator / (nominator + additional_denominator)
    return proportion


def select_threshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers
    """
    best_epi = 0
    best_F1 = 0

    step_size = (max(pval) - min(pval)) / 1000
    epi_range = np.arange(pval.min(), pval.max(), step_size)
    '''iteratively search for the best epsilon with respect to the F-score'''
    for epi in epi_range:
        '''true positive. false positive, false negative'''
        '''the label y = 1 corresponds to an anomalous example, and y = 0 corresponds to a normal example'''
        predictions = (pval < epi)[:, np.newaxis]
        tp = np.sum(predictions[yval == 1] == 1)
        fp = np.sum(predictions[yval == 0] == 1)
        fn = np.sum(predictions[yval == 1] == 0)

        '''compute precision, recall and F1'''
        prec = calc_proportion(tp, fp)
        rec = calc_proportion(tp, fn)

        if prec+rec == 0:
            F1 = 0
        else:
            F1 = (2 * prec * rec) / (prec + rec)
        '''update best_F1 and best_epi if the F-score improves'''
        if F1 > best_F1:
            best_F1 = F1
            best_epi = epi

    return best_epi, best_F1


def plot_optimal_threshold(X, epsilon, p, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    outliers = np.nonzero(p < epsilon)[0]
    plt.scatter(X[outliers, 0], X[outliers, 1], marker="o", facecolor="none", edgecolor="r", s=70)
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.title(f"{title}")
    plt.grid()
    plt.show()


def split_train_cv_test_normal_and_anomaly(mat):
    """
    this function takes an ndarray of the data, splits the labels to normal and non-normal, and splits the data so
    that the training sample gets 60% of the normal samples, and cv and test get 20% of normals,
    and 50% of the anomaly samples
    :param mat: dict data with target
    :return: split data samples: X_train, X_cv, X_test, y_train, y_cv, y_test
    """

    '''create a feature dataset and a label dataset'''
    X = mat["Xval"]
    y = mat["yval"]
    '''distinguish normal and anomaly samples'''
    normal = np.squeeze(mat["yval"] == 0)
    anomaly = np.squeeze(mat["yval"] == 1)
    X_normal = X[normal]
    X_anomaly = X[anomaly]
    y_normal = y[normal]
    y_anomaly = y[anomaly]
    '''randomly split the samples of normal and anomaly between training CV and test'''
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


def main(path_in, file_name):
    '''get the data'''
    X, yval, mat = get_the_data(path_in, file_name)
    plot_the_data(X, 'The Data')
    '''split the data to train, cross-validation and test. save all anomalies to cross validation and test periods'''
    X_train, X_cv, X_test, y_train, y_cv, y_test = split_train_cv_test_normal_and_anomaly(mat)
    '''estimate parameters (mean and variance) for the Gaussian model for training only'''
    mu, sigma2 = estimate_gaussian(X_train)
    '''for every sample compute its product of probability-density-functions over all the features 
    The low probability examples are more likely to be the anomalies in our dataset.'''
    pval = multivariate_gaussian(X_cv, mu, sigma2)
    '''find the best threshold (epsilon) using F-score in cross-validation.'''
    epsilon, F1 = select_threshold(y_cv, pval)
    print("Best epsilon found using cross-validation:", epsilon)
    print("Best F1 on Cross Validation Set:", F1)
    '''Visualizing the optimal threshold cv'''
    plot_optimal_threshold(X_cv, epsilon, pval, title="Cross Validation")
    '''compute the probabilities for the test sample'''
    pval = multivariate_gaussian(X_test, mu, sigma2)
    '''Visualizing the optimal threshold test'''
    plot_optimal_threshold(X_test, epsilon, pval, title="Test")


if __name__ == '__main__':
    path_in = config_param_path_in()
    file_name = r'\ex8data1.mat'
    main(path_in, file_name)