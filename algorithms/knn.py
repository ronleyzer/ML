from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print("\n ***** KNN with the Iris flower data exercises ***** ")
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
    '''evaluate different K options'''
    k_neighbors_options = list(range(3, 8))
    train_accuracy = []
    test_accuracy = []
    for k in k_neighbors_options:
        '''create a model'''
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        y_pred = knn.predict(X_test)
        '''Compute accuracy on the training set'''
        train_accuracy.append(knn.score(X_train, y_train))
        '''Compute accuracy on the testing set'''
        test_accuracy.append(knn.score(X_test, y_test))

    '''Visualization of k values vs accuracy'''
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(k_neighbors_options, test_accuracy, label='Testing Accuracy')
    plt.plot(k_neighbors_options, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    # plt.close()