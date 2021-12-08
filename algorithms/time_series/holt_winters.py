import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def background():
    print('''Holt-Winters Exponential Smoothening (HWES) works on the idea of smoothening the values of a 
             Univariate Time Series Analysis to use them for forecasting future values. 
             The idea is to assign exponentially decreasing weights giving more importance to more recent incidents. 
             So, when we move back in time, we would see diminishing weights.''')


def main():
    '''
    This code is an implementation of Holt-Winters forecasting
    sources: https://medium.com/analytics-vidhya/python-code-on-holt-winters-forecasting-3843808a9873
            https://medium.com/analytics-vidhya/holt-winters-forecasting-13c2e60d983f
    '''

    background()

    '''get the data'''
    airline = pd.read_csv(r'P:\ML\data\international-airline-passengers.csv', index_col='Month', parse_dates=True)
    print(airline.shape)

    '''plotting the original data'''
    airline[['Thousands of Passengers']].plot(title='Passengers Data')

    '''decomposing the Time Series
    Look for: Levels, Trends and Seasonality in the data'''
    decompose_result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')
    decompose_result.plot()
    '''We can quite clearly see that the data has all 3, Levels, Trends, Seasonality.'''

    '''Fitting the Data with Holt-Winters Exponential Smoothing'''

    print("HY")

if __name__ == '__main__':
    main()