import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
from visualization import scatter_plot_with_text


def background():
    print('''Non-negative matrix factorization (NMF or NNMF) 
    Is a dimensionality reduction method that is widely used in image processing, text mining, and more.
    A matrix V (Visible) is factorized into (usually) two matrices W (Weights) and H (Hidden), 
    with the property that all three matrices have no negative elements. 
    This non-negativity makes the resulting matrices easier to inspect.
    
    For a matrix V of dimensions m x n, 
    where each element is ≥ 0, NMF can factorize it into two matrices W 
    and H having dimensions m x k and k x n respectively and these two matrices only contain non-negative elements.
    Here, matrix V is defined as: V = WH, where,
    V -> Original Input Matrix (Linear combination of W & H) ; m*n
    W -> Feature Matrix ; m*k
    H -> Coefficient Matrix (Weights associated with W) ; k*n
    k -> Low rank approximation of V (k ≤ min(m,n))
    
    Sources: https://www.geeksforgeeks.org/non-negative-matrix-factorization/
             https://python.plainenglish.io/non-negative-matrix-factorization-for-dimensionality-reduction-predictive-hacks-1ed9e91154c
             https://towardsdatascience.com/nmf-a-visual-explainer-and-python-implementation-7ecdd73491f8''')


def main():
    background()

    '''get the eurovision data data'''
    path_in = r'P:\ML\data'
    eurovision = pd.read_csv(fr"{path_in}\eurovision-2016.csv")

    '''clean the data'''
    televote_Rank = eurovision.pivot(index='From country', columns='To country', values='Televote Rank')
    '''fill NAs by min per country'''
    televote_Rank.fillna(televote_Rank.min(), inplace=True)
    print("Raw data shape: ", televote_Rank.shape)

    '''Since we have the data in the right form, we are ready to run the NNMF algorithm.
    We will choose two components because our goal is to reduce the dimensions into 2.'''
    '''Create an NMF model'''
    model = NMF(n_components=2, max_iter=1000, random_state=0, init='random')
    model.fit(televote_Rank)
    nmf_features = model.transform(televote_Rank)
    '''Print the NMF features'''
    print("Features shape: ", nmf_features.shape)
    print("Components shape: ", model.components_.shape)
    print("we created two matrices Features and Components, of (42,2) and (2,26) dimensions respectively.")

    '''Plot the 42 Countries in two Dimensions'''
    text = np.array(televote_Rank.index)
    scatter_plot_with_text(df=nmf_features, text=text, font_size=10,
                           title='Dimensionality Reduction in Eurovision Data',
                           ylabel='Dimension1', xlabel='Dimension0')
    '''The 2D figure show patterns in similar countries such as Yugoslavians, Baltic, Scandinavian 
    and the countries of the United Kingdom.'''

    '''use Iris data'''
    data = load_iris()
    nmf = NMF(n_components=2, max_iter=1000, random_state=0, init='random', l1_ratio=0.1)
    x = nmf.fit_transform(data)
    print(nmf.inverse_transform(x[0]))


if __name__ == '__main__':
    main()