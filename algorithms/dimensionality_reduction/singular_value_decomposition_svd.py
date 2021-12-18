import numpy as np
from sklearn.decomposition import TruncatedSVD
from numpy import linalg as LA
from numpy.linalg import eig

def background():
    print('''Singular Value Decomposition (SVD) is a dimensionality reduction method.
    SVD allow to represent a large matrix A by 3 smaller matrices U, Σ and V.
    A_(m×n)= U_(m×m) Σ_(m×n) 〖V^T〗_(n×n)

    The rank of a matrix A is the maximum number of linearly independent column vectors in the matrix (p<n). 
    The rank of a matrix can be thought of as a representative of the amount of unique information represented 
    by the matrix. Higher the rank, higher the information. It is possible to lower the dimensionality of A by:
    a. Reduce the columns that are linearly dependent
    b. Reduce the columns that are close linearly dependent. 
       In a practical application, only the first few singular values Σ are large. 
       The rest of the singular values approach zero that can be ignored without losing much of the information.

    If the dimensions of A are m x n:
    U is an m x m matrix, also cold the Left Singular Vectors
    Σ  is an m x n rectangular diagonal matrix also cold the Singular Values, arranged in decreasing order
    V is an n x n matrix also cold the Right Singular Vectors
    U and V are both going to be orthonormal matrices: V^T V=I_p  and U^T U=I_p where I_p  is the identity matrix
    
    SVD is used in image compression, image recovery, eigenfaces, spectral clustering and more.

    Sources: https://www.analyticsvidhya.com/blog/2019/08/5-applications-singular-value-decomposition-svd-data-science/,
             https://www.youtube.com/watch?v=HAJey9-Q8js
''')


def main():
    '''*** simple implementation ***'''
    '''create the basic matrix'''
    A = np.array([[-1, 2, 0], [2, 0, -2], [0, -2, 1]])
    print("Original Matrix:")
    print(A)

    '''In order to reduce the dimantions we need to find all zeros or close to zeros singular values.
    Singular values of matrix A = eigenvalues of matrix A.T*A'''
    A_2 = (A.T).dot(A)
    eigenvalues_A2, eigenvectors_A2 = eig(A_2)
    print("The eigen values of A transpose A:")
    print(eigenvalues_A2)
    singular_values_A = np.sqrt(eigenvalues_A2)
    print("The singular values of matrix A:")
    print(singular_values_A)
    print('''python have numeric problem, the eigenvalues of matrix A.T*A are 9,0,9
    therefore the singular values of matrix A are 3,0,3. The meaning is that the second singular value
    is linearly dependent with another column and can be reduce without a great loss of information.''')

    '''create the model'''
    '''In TruncatedSVD class can be created in which you must specify the number 
    of desirable features or components to select, e.g. 2. '''
    svd = TruncatedSVD(n_components=2)
    svd.fit_transform(A)
    print("Singular values:")
    print(svd.singular_values_)
    print("explained variance:")
    print(svd.explained_variance_ratio_)
    print("sum of explained variance:")
    print(svd.explained_variance_ratio_.sum())
    print("Transformed Matrix after reducing to 2 features:")
    print(svd.fit_transform(A))


if __name__ == '__main__':
    background()
    main()