from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

    print("DAMN!")