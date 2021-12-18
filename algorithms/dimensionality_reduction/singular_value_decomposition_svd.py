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
    background()


if __name__ == '__main__':
    main()