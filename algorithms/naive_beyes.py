# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import numpy as np
from sklearn.svm._libsvm import predict_proba


def separate_by_class(dataset):
    '''
    Split the dataset by class values
    :param dataset: df
    :return: returns a dictionary
    '''
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


def split_data_to_x_y(df):
    class_value = []
    for i in range(len(df)):
        vector = df[i]
        class_value.append(vector[-1])
        df[i] = vector[:-1]
    return df, class_value


def summarize_dataset(dataset):
    '''
    :param dataset: df
    :return: mean, stdev and count for each column in a dataset
    '''
    summaries = dict()
    summaries['mean'] = [np.mean(column) for column in zip(*dataset)]
    summaries['std'] = [np.std(column) for column in zip(*dataset)]
    summaries['count'] = [len(column) for column in zip(*dataset)]
    return summaries


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
summ = summarize_dataset(X_train)
gnb = GaussianNB()
# cat = CategoricalNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
train_accuracy = gnb.score(X_train, y_train)
test_accuracy = gnb.score(X_test, y_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

'''new data set'''
dataset = [
            [3.393533211, 2.331273381, 0],
            [3.110073483, 1.781539638, 0],
            [1.343808831, 3.368360954, 0],
            [3.582294042, 4.67917911, 0],
            [2.280362439, 2.866990263, 0],
            [7.423436942, 4.696522875, 1],
            [5.745051997, 3.533989803, 1],
            [9.172168622, 2.511101045, 1],
            [7.792783481, 3.424088941, 1],
            [7.939820817, 0.791637231, 1]
        ]
separated = separate_by_class(dataset)
X, y = split_data_to_x_y(dataset)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
cat = CategoricalNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

