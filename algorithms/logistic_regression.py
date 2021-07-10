from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


if __name__ == '__main__':
    print("\n ***** logistic regression with the Iris flower data ***** ")
    ''' read the data '''
    X, y = load_iris(return_X_y=True)
    df = pd.DataFrame(X)
    df['y'] = y.tolist()
    ''' describe the data '''
    print("\ninfo:", df.info())
    print("\ndescribe:\n", df.describe())
    print("\nvalue_counts:\n", df['y'].value_counts())
    '''describe by label of y'''
    for label in np.unique(df['y']):
        print(f"\nDescribe X when y = {label}\n", df[df['y'] == label].describe())
    '''split the data'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    ''' model '''
    logistic_regression_model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = logistic_regression_model.predict(X_test)
    y_prob = logistic_regression_model.predict_proba(X_test)
    logistic_regression_model.score(X, y)
    '''Compute accuracy on the training set'''
    print("\nlogistic regression train accuracy:", logistic_regression_model.score(X_train, y_train))
    '''Compute accuracy on the testing set'''
    print("\nlogistic regression test accuracy:", logistic_regression_model.score(X_test, y_test))