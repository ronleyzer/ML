import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

sys.path.append(os.getcwd())
from generic_fun.get_data import config_param_path_in

'''the code is based on https://www.kaggle.com/kevinarvai/outlier-detection-practice-uni-multivariate'''


def isolate_forest_simple(df):
    for i, column in enumerate(df.columns.tolist()):
        isolation_forest = IsolationForest(contamination='auto', random_state=111)
        isolation_forest.fit(df[column].values.reshape(-1, 1))
    return isolation_forest.predict(df[column].values.reshape(-1, 1))


def isolate_forest(df, subplots_rows):
    """
    The function plots the isolation forest scores of the data.

    The Isolation Forest score is the anomaly score of each sample.
    The IsolationForest 'isolates' observations by randomly selecting a feature and then randomly selecting a
    split value between the maximum and minimum values of the selected feature. The number of splittings required
    to isolate a sample is equivalent to the path length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a measure of normality. When a forest of random
    trees collectively produces shorter path lengths for particular samples, they are highly likely to be anomalies.
    We set 0 as the threshold value for outliers."""

    '''plot related functionality'''
    fig, axs = plt.subplots(subplots_rows, len(df.columns), figsize=(14, 8), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    fig.suptitle("Isolation Forest - Anomaly Score")

    '''give each sample an anomaly score, and determine if it is an outlier (-1 for outlier, else 1)'''
    cols = df.columns.tolist()
    outliers = {}
    anomaly_scores = {}
    for i, column in enumerate(cols):
        isolation_forest = IsolationForest(random_state=111)
        isolation_forest.fit(df[column].values.reshape(-1, 1))
        xx = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1, 1)
        anomaly_score = isolation_forest.decision_function(xx)
        outlier = isolation_forest.predict(xx)

        outliers[column] = outlier
        anomaly_scores[column] = anomaly_score

        axs[i].plot(xx, anomaly_score, label='anomaly score')
        axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                            where=outlier == -1, color='r',
                            alpha=.4, label='outlier region')
        axs[i].legend()
        axs[i].set_title(column)
    plt.tight_layout()
    fig.tight_layout()
    plt.show()


def main(path_in, file_name):
    'get the data'
    df = pd.read_csv(fr'{path_in}\{file_name}', index_col=[0], parse_dates=[0], dayfirst=True)
    'fill missing values'
    df.fillna(df.median(), inplace=True)
    'select only numeric data'
    df_num = df.select_dtypes(include=["float64", "int64"])
    'visualize histogram'
    df_num[df_num.columns.tolist()].hist(figsize=(15, 8))
    plt.suptitle('Feature Histograms')
    plt.tight_layout()
    plt.show()
    'select the subset of columns for which we detect outliers'
    cols = df_num.columns.to_list()[1:4]
    df_subset = df_num[cols]
    'run isolate forest'
    isolate_forest(df_subset, subplots_rows=1)


if __name__ == '__main__':
    path_in = config_param_path_in()
    file_name = r'\anomaly_detection\Melbourne_housing_FULL.csv'
    main(path_in, file_name)