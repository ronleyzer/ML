from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

if __name__ == '__main__':
    print("\n ***** KFold with the Iris flower data exercises ***** ")
    ''' read the data '''
    X, y = load_iris(return_X_y=True)
    df = pd.DataFrame(X)
    df['y'] = y.tolist()

    '''split the data train test vs kfold'''
    kf = KFold(n_splits=3, random_state=1, shuffle=True)
    model = RandomForestClassifier()
    i = 1
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model
        model.fit(X_train, y_train)  # Training the model
        print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, model.predict(X_test))}")
        i += 1

        '''split the data RepeatedKFold vs RepeatedStratifiedKFold'''
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6]])
        random_state = 12883823
        rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
        print("RepeatedKFold:")
        for train, test in rkf.split(X):
            print("%s %s" % (train, test))

        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        random_state = 12883823
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=random_state)
        print("RepeatedStratifiedKFold:")
        for train2, test2 in rskf.split(X, y):
            print("%s %s" % (train2, test2))
