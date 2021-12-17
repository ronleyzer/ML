import glob
import cv2
import numpy as np


def ica(X, step_size=1, tol=1e-8, max_iter=10000, n_sources=4):
    """
    Perform ICA
    :param X: Input dataset, numpy array of dimension (n_samples, n_features)
    :param step_size: int that controls step size for gradient descent
    :param tol: Tolerance for limit condition
    :param max_iter: Maximum number of iterations
    :param n_sources: Number of mixed sources
    :return: Weight matrix for the mixed signal, numpy array of dimension (n_sources, n_samples)
    """
    m, n = X.shape

    # Initialize random weights
    W = np.random.rand(n_sources, m)

    for c in range(n_sources):
        # Copy weights associated to the component and normalize
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())

        for i in range(max_iter):
            # Dot product of weight and input
            v = np.dot(w.T, X)

            # Pass w*s into contrast function g
            gv = np.tanh(v*step_size).T

            # Pass w*s into g prime
            gdv = (1 - np.square(np.tanh(v))) * step_size

            # Update weights
            wNew = (X * gv.T).mean(axis=1) - gdv.mean() * w.squeeze()

            # Decorrelate and normalize weights
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)
            # Update weights
            w = wNew
            # Check for convergence
            if lim < tol:
                break

        W[c, :] = w.T
    return W


def nmf(X, tol=1e-6, max_iter=5000, n_components=4):
    """
    Perform NMF
    :param X: Input dataset, numpy array of dimension (n_samples, n_features)
    :param tol: Tolerance for limit condition
    :param max_iter: Maximum number of iterations
    :param n_components: Number of mixed sources
    :return: Components matrix for the mixed signal sources, numpy array of dimension (n_sources, n_features)
    """

    '''To apply NMF, we start by initializing W and H randomly.'''
    # Random initialization of weights
    W = np.random.rand(X.shape[0], n_components)
    H = np.random.rand(n_components, X.shape[1])

    # A very big number for initial error
    oldlim = 1e9

    # A very small number to ensure matrices are strictly positive
    eps = 1e-7

    '''Then, at each iteration, we first update new H given old H and W,
    then new W given H and old W, using multiplicative update rules until convergence.
    We use the Frobenius norm between X and WH to check for convergence.'''
    for i in range(max_iter):
        # Multiplicative update steps
        H = H * ((W.T.dot(X) + eps) / (W.T.dot(W).dot(H) + eps))
        W = W * ((X.dot(H.T) + eps) / (W.dot(H.dot(H.T)) + eps))

        # Frobenius distance between W•H and X
        lim = np.linalg.norm(X-W.dot(H), 'fro')

        # Check for convergence
        if abs(oldlim - lim) < tol:
            break

        oldlim = lim

    '''Finally, we normalize the H matrix obtained, and reshape it to have it contain
    2D images before saving visualizations.

    As we can see above, the NMF algorithm works considerably better than ICA for our example.'''
    for j in range(n_components):
        # Normalize each source image
        H[j,:] *= 255.0 / np.max(H[j,:])
    return H


def main():
    '''source: https://towardsdatascience.com/step-by-step-signal-processing-with-machine-learning-pca-ica-nmf-8de2f375c422'''
    images = []
    path_in = r'P:\ML\data'
    path_out = r'P:\ML\data\output'
    for path in glob.glob(fr"{path_in}\mixture_dataset(0147)/*.jpg"):
        images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))

    # Reshape images to 1D arrays
    flat_images = np.array(images).reshape(856, -1)

    # Obtain weights matrix from ICA
    W = ica(flat_images, step_size=1.5)

    # Recover original sources
    S = W.dot(flat_images).reshape(-1, 28, 28)
    for i, im in enumerate(S):
        cv2.imwrite("output/ica_{}.jpg".format(i), im)

    H = nmf(flat_images).reshape(-1, 28, 28)
    for i, im in enumerate(H):
        cv2.imwrite("output/nmf_{}.jpg".format(i), im)


if __name__ == "__main__":
    main()