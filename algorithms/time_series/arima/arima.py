import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

'''
the code is based on
https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/ ,
https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
'''


def find_non_stationary_and_plot(ts):
    """
    Plot the ts features and their respective Adfuller P-values to check for stationarity
    """
    for feature in ts.columns:
        result = adfuller(ts[feature].dropna())
        pd.Series(ts[feature]).plot()
        plt.suptitle('Time Series Data', fontsize=18)
        plt.title(f'Adfuller P-Val={round(result[1], 3)}')
        plt.grid()
        plt.show()
        plt.close()


def get_the_data(path):
    y = pd.read_csv(path, names=['value'], header=0)
    return y


def plot_series_and_autocorrelation(axes, ts, index, title):
    axes[index, 0].plot(ts)
    result = adfuller(ts.dropna())
    axes[index, 0].set_title(f'{title}\nAdfuller P-value: {np.round(result[1], 3)}')
    plot_acf(ts.dropna(), ax=axes[index, 1], lags=np.arange(1, ts.shape[0]-index))
    axes[index, 0].grid()


def examine_auto_correlation(ts):
    """
    plots the original series and its first and second differencing as well as its autocorrection.

    If the auto correlations are positive for many number of lags (10 or more),
    then the series needs further differencing.
    On the other hand, if the lag 1 auto correlation itself is too negative,
    then the series is probably over-differentiated.
    """
    fig, axes = plt.subplots(3, 2, sharex=True)
    plot_series_and_autocorrelation(axes, ts.value, 0, title='Original Series')
    plot_series_and_autocorrelation(axes, ts.value.diff(), 1, title='1st Order Differencing')
    plot_series_and_autocorrelation(axes, ts.value.diff().diff(), 2, title='2nd Order Differencing')
    plt.tight_layout()
    plt.show()


def acf(series, acf, ylim, title):
    '''PACF plot difference series'''
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(series)
    axes[0].set_title(title)
    axes[1].set(ylim=ylim)
    if acf == 'partial':
        plot_pacf(series.dropna(), ax=axes[1], lags=np.arange(1, np.round((series.shape[0])/2-1, 0)))
        # -1 for first diff, -2 for 2nd diff
    else:
        plot_acf(series.dropna(), ax=axes[1], lags=np.arange(1, (series.shape[0]-1)))
    plt.show()


def arima_model(ts, order):
    model = ARIMA(endog=ts.value, exog=None, order=order)
    model_fit = model.fit()
    return model_fit


def residual_plot(model_fit):
    """Plot residual errors"""
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.iloc[1:, :].plot(title="Residuals", ax=ax[0])
    residuals.iloc[1:, :].plot(kind='kde', title='Density', ax=ax[1])
    plt.show()


def actual_vs_predict_plot(model_fit, ts_validation, steps, alpha, order):
    '''Actual vs Fitted'''
    '''forecast'''
    '''add the last sample of the validation to train - avoid gap'''
    model_fit.data.endog = np.append(model_fit.data.endog, ts_validation.iloc[0, 0])
    forecast = model_fit.forecast(steps=steps, alpha=alpha)
    fc_series = pd.Series(forecast, index=ts_validation.index).fillna(model_fit.data.endog[-1])
    plt.plot(np.squeeze(fc_series), label='forecast', color='r')
    plt.plot(model_fit.data.endog, label='training')
    plt.plot(ts_validation, label='actual')
    plt.title(f'Forecast vs Actual ARIMA\nparams: AR={order[0]}, target_diff={order[1]}, MA={order[1]}')
    plt.grid()
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


def split_time_series_to_train_validation_test(ts, split_pct):
    number_of_samples = len(ts)
    num_of_observation_in_train = int(np.round(split_pct['train']*number_of_samples, 0))
    num_of_observation_in_validation = int(np.round(split_pct['validation']*number_of_samples, 0))
    '''create training and test'''
    train = pd.DataFrame(ts[:(num_of_observation_in_train)])
    validation = pd.DataFrame(ts[num_of_observation_in_train:
                                       (num_of_observation_in_train+num_of_observation_in_validation)])
    test = pd.DataFrame(ts[(num_of_observation_in_train+num_of_observation_in_validation):])
    return train, validation, test


def auto_correlation(selected_diff):
    autocorrelation_plot(selected_diff.dropna())
    plt.title('AR Selection Using Auto-Correlation')
    plt.show()


def main():
    '''get the data'''
    ts = get_the_data(path='https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv')
    split_pct = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
    ts, ts_validation, ts_test = split_time_series_to_train_validation_test(ts, split_pct)

    '''A. test if the series is stationary, and iteratively differentiate the series till it is stationary.
    the null hypothesis of the adfuller test is that the time series is non-stationary. 
    therefore, if the p-value of the test is less than the significance level (0.05) the null hypothesis is rejected.
    A.1 test stat'''
    find_non_stationary_and_plot(ts)

    '''A.2. select number of diffs-
    since the P-value is greater than the significance level, 
    let’s difference the series and see how the auto correlation plot looks.'''
    examine_auto_correlation(ts)
    '''The time series reaches stationarity with two orders of differencing, 
    but at this stage the autocorrelation function becomes not smooth.
    This indicates that the series was over differenced.'''

    '''B. Find the order of the AR term (param p)-
    find out the required number of AR terms using the Partial Auto correlation (PACF) plot.
    Partial auto correlation can be imagined as the correlation between the series and its lag, 
    after excluding the contributions from the intermediate lags.
    '''
    selected_diff = ts.value.diff()
    '''PACF expresses the correlation between observations made at two points in time while accounting 
    for any influence from other data points. We can use PACF to determine the optimal number of terms to use 
    in the AR model. The number of terms determines the order of the model.'''
    auto_correlation(selected_diff)
    '''Both PACF lag 1 and 2 are above the significance line.'''

    '''C. Find the order of the MA term (q)- An MA term is the error of the lagged forecast.
    The ACF tells how many MA terms are required to remove any auto correlation in the stationary series.
    Auto Correlation Function (ACF)- The correlation between the observations at the current point in time 
    and the observations at all previous points in time. We can use ACF to determine the optimal number of MA terms. 
    The number of terms determines the order of the model.'''

    '''D. After determining the values of p, d and q, fit the ARIMA model.'''
    model_fit = arima_model(ts, order=(1, 1, 2))
    print('''\nNotice that the coefficient of the MA2 term is close to zero and the P-Value in the ‘P>|z|’ 
    column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.
    Let’s rebuild the model without the MA2 term.\n''')
    print(model_fit.summary())
    order = (1, 1, 1)
    model_fit = arima_model(ts, order=order)
    print('''\nThe P Values of the AR1 and MA1 terms improved and are significant (<< 0.05)..\n''')
    print(model_fit.summary())

    '''E. Forecast validation and plot the actual against the fitted values'''
    actual_vs_predict_plot(model_fit, ts_validation, steps=len(ts_validation), alpha=0.05, order=order)
    '''From the chart, the ARIMA(1,1,1) model seems to give a directionally correct forecast. 
    But each of the predicted forecasts is consistently below the actual. 
    This means that by adding a small constant to our forecast, the accuracy will certainly improve.'''

    '''F. Manually correcting the orders - increase the order of differencing to two, 
    that is set d=2 and iteratively increase p to up to 5 and then q up to 5 to 
    check which model gives least AIC, and look for a chart that gives closer actual and forecasts.
    While doing that the P values of the AR and MA terms should be as close to zero, ideally, less than 0.05.'''
    orders = [(2, 1, 1), (1, 1, 2), (3, 1, 2), (3, 1, 1)]
    for order in orders:
        model_fit = arima_model(ts, order=order)
        actual_vs_predict_plot(model_fit, ts_validation, steps=len(ts_validation), alpha=0.05, order=order)

    '''G. Plot the residuals to ensure that there are no patterns (that is, look for constant mean and variance)'''
    residual_plot(model_fit)
    '''The residual errors seem fine with near zero mean and uniform variance.'''
    print(model_fit.summary())


if __name__ == '__main__':
    main()