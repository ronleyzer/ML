import numpy as np
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import eig


def background():
    print('''Singular Value Decomposition (SVD) is a dimensionality reduction method. 
    SVD allows to represent a matrix A by 3 matrices U, Σ and V.
    A_(m×n)= U_(m×m) Σ_(m×n) 〖V^T〗_(n×n)

    Where:
    U is an m x m matrix, also called the Left Singular Vectors. 
      The left singular vectors of matrix A are the eigenvectors of the matrix 〖AA〗^T .  
    Σ is an m x n rectangular diagonal matrix also called the Singular Values, arranged in decreasing order. 
      The singular values of matrix A are the square root of the eigenvalues of the matrix A^T A .  
    V is an n x n matrix also called the Right Singular Vectors. The right singular vectors of matrix A 
      are the eigenvectors of the matrix A^T A .  
    U and V are both orthonormal matrices: V^T V=I_p  and U^T U=I_p 

    The rank of a matrix A is the maximum number of linearly independent column vectors in the matrix (p<n). 
    The rank of a matrix can be thought of as a representative of the amount of unique information represented by 
    the matrix. The higher the rank, the higher the information. It is possible to lower the dimensionality of A by:
    a. Reduce the columns that are linearly dependent.
    b. Reduce the columns that are close to linearly dependent. In practical applications, only the first few singular 
       values of Σ are large. The rest of the singular values approach zero and can be ignored without losing much of 
       the information.
    
    SVD is used in image compression, image recovery, eigenfaces, spectral clustering and more.
    Sources: https://www.analyticsvidhya.com/blog/2019/08/5-applications-singular-value-decomposition-svd-data-science/,
             https://www.youtube.com/watch?v=HAJey9-Q8js''')


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
    print('''python have numeric problem, The eigen values of A^T A are [ 9,0,9]. 
             Therefore, the singular values of matrix A are [3,0,3]. This means that one of the singular vectors 
             is in the null space of A^T A, and it can be reduced without a loss of information.''')

    '''create the model'''
    '''Using TruncatedSVD must specify the number of desirable features or components to select, e.g. 2. '''
    svd = TruncatedSVD(n_components=2)
    svd.fit_transform(A)
    print("Singular values:")
    print(svd.singular_values_)
    print("explained variance:")
    print(svd.explained_variance_ratio_)
    print("sum of explained variance:")
    print(svd.explained_variance_ratio_.sum())
    print("The reduced matrix after reducing to 2 features is:")
    print(svd.fit_transform(A))
    '''After fitting and transforming the SVD model with only 2 components, 
    we have 2 remaining singular values: [3. 3.], their explained variance is: [0.48846154 0.51153846].
    The sum of explained variance is 1, so the information is kept.'''


if __name__ == '__main__':
    background()
    main()