from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from scipy import sparse
import sklearn.datasets as ds

class prediction_models(object):
    def __init__(self, X, y, new_xs):
        self.X = X
        self.y = y
        self.new_xs = new_xs

    def knn_classifier(self, k_neighbors):
        neigh = KNeighborsClassifier(n_neighbors=k_neighbors)
        neigh.fit(X, y)
        KNeighborsClassifier(...)
        for new_x in new_xs:
            print('new_x: ', new_x)
            print('prediction:', neigh.predict([[new_x]]))
            print('probability: ', neigh.predict_proba([[new_x]]))

    def knn_regressor(self, k_neighbors):
        neigh = KNeighborsRegressor(n_neighbors=k_neighbors, weights='distance')
        neigh.fit(X, y)
        KNeighborsClassifier(...)
        for new_x in new_xs:
            print('new_x: ', new_x)
            print('prediction:', neigh.predict([[new_x]]))


if __name__ == '__main__':

    X = [[0], [1], [2], [3], [7], [7]]
    y = [0, 0, 1, 1, 1, 1]
    new_xs = [1.6]
    k_neighbors = 2

    knn_model = prediction_models(X, y, new_xs)
    print("\nknn_classifier - ")
    knn_model.knn_classifier(k_neighbors)
    print("\nknn_regressor - ")
    knn_model.knn_regressor(k_neighbors)

    print("\n\n\n ***** The Iris flower data exercises ***** ")
    '''if working with the data as np array : '''
    iris = ds.load_iris()
    # research the data
    '''if working with the data as np array : '''
    print("\niris.keys: ", iris.keys())
    X, y = iris.data, iris.target
    print("\nX shape: ", X.shape)
    print("X mean: ", X.mean(axis=0))
    print("X min: ", X.min(axis=0))
    print("X max: ", X.max(axis=0))

    print("\ny shape: ", y.shape)
    print("y mean: ", y.mean(axis=0))
    print("y min: ", y.min(axis=0))
    print("y max: ", y.max(axis=0), "\n")

    '''if working with the data as dataframe : '''
    # filename = iris.filename
    # iris_0 = pd.read_csv(filename)
    iris_data = pd.DataFrame(X)
    iris_data['y'] = y.tolist()
    print("\ninfo:", iris_data.info())
    print("\ndescribe:\n", iris_data.describe())
    print("\nvalue_counts:\n", iris_data['y'].value_counts())
    # 2-D array with ones on the diagonal and zeros elsewhere
    # matrix = np.zeros((3, 3))
    # diag_of_ones = np.ones((1, 3))
    # matrix[np.diag_indices(3, ndim=2)] = diag_of_ones
    eye = np.eye(4)
    # convert the NumPy array to a SciPy sparse matrix in CSR format.
    sparse_matrix = sparse.csr_matrix(eye)


    print("DAMN!")

