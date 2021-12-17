from sklearn.decomposition import FastICA
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile


def background():
    print('''Independent Component Analysis (ICA)\n
            Independent component analysis separates a multivariate signal into additive subcomponents that are
            maximally independent. The method assume that the subcomponents are non-Gaussian signals and that they
            are statistically independent from each other. Example application is the "cocktail party problem" 
            of listening in on one person's speech in a noisy room.

            
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


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        X += 0.02 * np.random.normal(size=X.shape)

    return X


def main():
    background()

    '''create the data'''
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

    ''''''
    path_in = r'P:\ML\data\ica'
    sampling_rate, mix1 = wavfile.read(fr'{path_in}\mix1.wav')
    sampling_rate, mix2 = wavfile.read(fr'{path_in}\mix2.wav')
    sampling_rate, source1 = wavfile.read(fr'{path_in}\source1.wav')
    sampling_rate, source2 = wavfile.read(fr'{path_in}\source2.wav')

    S = np.c_[source1, source2]
    X = mix_sources([mix1, mix2]).T
    ica = FastICA(n_components=2)
    S_ = ica.fit_transform(X)
    ica_plot(X, S, S_)


if __name__ == '__main__':
    main()