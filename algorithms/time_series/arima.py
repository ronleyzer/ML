import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})


def find_non_stationary_and_plot(df):
    non_stationary = []
    for feature in df.columns:
        """
        The following commented out lines show how to deal with cases in which addfuller fails because 
        of to many values to unpack- use np.squeeze()- same as in the line of check2
        """
        result = adfuller(df[feature].dropna())
        pd.Series(df[feature]).plot(title=f'{feature}   P-Val={round(result[1], 3)}')
        plt.show()
        plt.close()
        if result[1] > 0.05:
            non_stationary.append(feature)
    return non_stationary, result


def get_the_data():
    y = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv',
                    names=['value'], header=0)
    return y


def examine_auto_correlation(df):
    '''
    If the auto correlations are positive for many number of lags (10 or more),
    then the series needs further differencing.
    On the other hand, if the lag 1 auto correlation itself is too negative,
    then the series is probably over-difference.

    :param df:
    :return: first and second difference of the y
    '''
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.value)
    result = adfuller(df.value.dropna())
    axes[0, 0].set_title(f'Original Series P-value: {np.round(result[1],3)}')
    plot_acf(df.value, ax=axes[0, 1], lags=np.arange(1, df.value.shape[0]))

    # 1st Differencing
    axes[1, 0].plot(df.value.diff())
    result = adfuller(df.value.diff().dropna())
    axes[1, 0].set_title(f'1st Order Differencing P-value: {np.round(result[1],3)}')
    plot_acf(df.value.diff().dropna(), ax=axes[1, 1], lags=np.arange(1, df.value.shape[0]))

    # 2nd Differencing
    axes[2, 0].plot(df.value.diff().diff())
    result = adfuller(df.value.diff().diff().dropna())
    axes[2, 0].set_title(f'2nd Order Differencing P-value: {np.round(result[1],3)}')
    plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1], lags=np.arange(1, df.value.shape[0]))

    plt.show()


def acf(series, acf, ylim, title):
    '''PACF plot of 1st difference series'''
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(series)
    axes[0].set_title(title)
    axes[1].set(ylim)
    if acf == 'partial':
        plot_pacf(series.dropna(), ax=axes[1])
    else:
        plot_pacf(series.dropna(), ax=axes[1])
    plt.show()


def main():
    '''
    ARIMA model
    based on: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    :return:
    '''
    df = get_the_data()
    '''A. test stationary and diff the series till it stat - 
    differencing needed only if the series is non-stationary
    null hypothesis of the adfuller test is that the time series is non-stationary. 
    So, if the p-value of the test is less than the significance level (0.05) 
    then you reject the null hypothesis and infer that the time series is indeed stationary.
    A.1 test stat'''
    non_stationary, result = find_non_stationary_and_plot(df)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    '''A.2. select number od diffs-
    Since P-value is greater than the significance level, 
    let’s difference the series and see how the auto correlation plot looks like.'''
    examine_auto_correlation(df)
    '''For the above series, the time series reaches stationary with two orders of differencing. 
    But on looking at the auto correlation plot for the 2nd differencing the lag goes into the 
    far negative zone fairly quick, which indicates, the series might have been over difference.'''

    '''B. Find the order of the AR term (p) -
    find out the required number of AR terms using Partial Auto correlation (PACF) plot.
    Partial auto correlation can be imagined as the correlation between the series and its lag, 
    after excluding the contributions from the intermediate lags.
    '''
    selected_diff = df.value.diff()
    acf(selected_diff, acf='partial', ylim=(0, 5), title='Order of AR Term')
    '''the PACF lag 1 is quite significant since is well above the significance line. 
    Lag 2 turns out to be significant as well, slightly managing to cross the significance limit (blue region). 
    But I am going to be conservative and tentatively fix the p as 1.'''

    '''C. Find the order of the MA term (q)
    An MA term is the error of the lagged forecast.
    The ACF tells how many MA terms are required to remove any auto correlation in the stationary series.'''
    acf(selected_diff, acf='', ylim=(0, 1.2), title='Order of MA Term')
    '''Couple of lags are well above the significance line. So, let’s tentatively fix q as 2. 
    When in doubt, go with the simpler model that sufficiently explains the Y.'''

    '''If your series is slightly under difference, adding one or more additional AR terms usually makes it up. 
    Likewise, if it is slightly over-difference, try adding an additional MA term.'''



    print("HY")


if __name__ == '__main__':
    main()