import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

sys.path.append(os.getcwd())
from generic_fun.get_data import config_param_path_in

'''Anomalies are patterns of different data within given data,
   whereas Outliers would be merely extreme data points within data. 
   the code used https://www.kaggle.com/kevinarvai/outlier-detection-practice-uni-multivariate code and data'''


def isolate_forest_simple(df):
    for i, column in enumerate(df.columns.tolist()):
        isolation_forest = IsolationForest(contamination='auto', random_state=111)
        isolation_forest.fit(df[column].values.reshape(-1, 1))
    return isolation_forest.predict(df[column].values.reshape(-1, 1))


def isolate_forest(df, subplots_rows, subplots_columns):
    fig, axs = plt.subplots(subplots_rows, subplots_columns, figsize=(22, 12), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    cols = df.columns.tolist()
    outliers = {}
    anomalies_score = {}
    for i, column in enumerate(cols):
        isolation_forest = IsolationForest(contamination='auto', random_state=111)
        isolation_forest.fit(df[column].values.reshape(-1, 1))
        xx = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1, 1)
        anomaly_score = isolation_forest.decision_function(xx)
        outlier = isolation_forest.predict(xx)

        outliers[column] = outlier
        anomalies_score[column] = anomaly_score

        axs[i].plot(xx, anomaly_score, label='anomaly score')
        axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                            where=outlier == -1, color='r',
                            alpha=.4, label='outlier region')
        axs[i].legend()
        axs[i].set_title(column)
    plt.show()
    return outliers, anomalies_score


def main(path_in, file_name):
    'Nonparametric methods: Univariate'
    'get the data'
    df = pd.read_csv(fr'{path_in}\{file_name}', index_col=[0], parse_dates=[0], dayfirst=True)
    'clean'
    df.fillna(df.median(), inplace=True)
    'select only numeric data'
    df_num = df.select_dtypes(include=["float64", "int64"])
    'visualize histogram'
    df_num[df_num.columns.tolist()].hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    'select subset of columns to detect outliers'
    cols = df_num.columns.to_list()[1:4]
    df_subset = df_num[cols]
    'isolate forest returns an anomaly_score for each data point'
    subplots_rows = 1
    subplots_columns = len(cols)
    outliers, anomalies_score = isolate_forest(df_subset, subplots_rows, subplots_columns)


if __name__ == '__main__':
    path_in = config_param_path_in()
    file_name = r'\anomaly_detection\Melbourne_housing_FULL.csv'
    main(path_in, file_name)