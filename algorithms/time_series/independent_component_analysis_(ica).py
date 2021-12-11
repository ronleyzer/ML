from sklearn.decomposition import FastICA
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def background():
    print('''Independent Component Analysis (ICA)\n
            Independent component analysis separates a multivariate signal into additive\n
            subcomponents that are maximally independent.\n\n
            For example:\n
            Suppose that you’re at a house party and you’re talking to some cute person.\n
            As you listen, your ears are being bombarded by the sound coming from the conversations going on between\n 
            different groups of people through out the house and from the music that’s playing rather loudly\n
            in the background. Yet, none of this prevents you from focusing in on what the person is saying since\n
            human beings possess the innate ability to differentiate between sounds.\n\n
            If, however, this were taking place as part of scene in a movie, the microphone which we’d use to record\n
            the conversation would lack the necessary capacity to differentiate between all the sounds going\n
            on in the room. This is where Independent Component Analysis, or ICA for short, comes in to play.\n
            ICA is a computational method for separating a multivariate signal into its underlying components.\n
            Using ICA, we can extract the desired component (i.e. conversation between you and the person) from the\n
            amalgamation of multiple signals.\n\n
            
            Independent Component Analysis (ICA) Algorithm\n
            At a high level, ICA can be broken down into the following steps:\n
            A. Center x by subtracting the mean\n
            B. Whiten x- whitening a signal involves the eigen-value decomposition of its covariance matrix.
                         preprocessing the signal, for each component.\n
            C. Choose a random initial value for the de-mixing matrix w\n
            D. Calculate the new value for w\n
            E. Normalize w\n
            F. Check whether algorithm has converged and if it hasn’t, return to step 4.
               Convergence is considered attained when the dot product of w and its transpose is roughly equal to 1.\n
            G. Take the dot product of w and x to get the independent source signals\n
            
            source: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
                    https://medium.com/analytics-vidhya/independent-component-analysis-for-signal-decomposition-3db954ffe8aa''')


def create_3_series_with_separate_patterns():
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * time)
    s2 = np.sign(np.sin(3 * time))
    s3 = signal.sawtooth(2 * np.pi * time)
    return s1, s2, s3


def ica_plot(X, S, S_):
    fig = plt.figure()
    models = [X, S, S_]
    names = ['mixtures', 'real sources', 'predicted sources']
    colors = ['red', 'blue', 'orange']
    for i, (name, model) in enumerate(zip(names, models)):
        plt.subplot(4, 1, i + 1)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)
    fig.tight_layout()
    plt.show()


def main():
    background()

    '''create the data or upload the data'''
    s1, s2, s3 = create_3_series_with_separate_patterns()
    ''' Mixing of Signals'''
    S = np.c_[s1, s2, s3]
    plt.plot(S)
    plt.show()

    '''Adding Noise '''
    S += 0.2 * np.random.normal(size=S.shape)
    '''Standardize the data'''
    S /= S.std(axis=0)
    plt.plot(S)
    plt.show()

    '''Take a basis vector which will decide the proportion of mixing of signals with each other.
    The observation mixture will be produced using dot product of basis vector A and signal mixture S'''
    '''Mixing the Data'''
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])

    '''Create the Observation Data for ICA'''
    X = np.dot(S, A.T)
    plt.plot(X)
    plt.show()

    '''Using this model after fit transforming will able to decompose the mixing matrix 
    and Signal mixture respectively A_ and S_'''
    ica = FastICA(n_components=3)
    '''Get Estimated Signals S_'''
    S_ = ica.fit_transform(X)

    '''Plotting the results'''
    ica_plot(X, S, S_)




if __name__ == '__main__':
    main()