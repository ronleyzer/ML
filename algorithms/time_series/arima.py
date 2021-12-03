import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
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


def get_the_data(path):
    y = pd.read_csv(path, names=['value'], header=0)
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
    plot_acf(df.value.diff().dropna(), ax=axes[1, 1], lags=np.arange(1, df.value.diff().shape[0]-1))

    # 2nd Differencing
    axes[2, 0].plot(df.value.diff().diff())
    result = adfuller(df.value.diff().diff().dropna())
    axes[2, 0].set_title(f'2nd Order Differencing P-value: {np.round(result[1],3)}')
    plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1], lags=np.arange(1, df.value.diff().diff().shape[0]-2))

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


def arima_model(df, order):
    model = ARIMA(endog=df.value, exog=None, order=order)
    model_fit = model.fit()
    return model_fit


def residual_plot(model_fit):
    '''Plot residual errors'''
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.iloc[1:, :].plot(title="Residuals", ax=ax[0])
    residuals.iloc[1:, :].plot(kind='kde', title='Density', ax=ax[1])
    plt.show()


def actual_vs_predict_plot(model_fit, df_validation, steps, alpha, order):
    '''Actual vs Fitted'''
    '''forecast'''
    forecast = model_fit.forecast(steps=steps, alpha=alpha)
    fc_series = pd.Series(forecast, index=df_validation.index).fillna(model_fit.data.endog[-1])
    plt.plot(np.squeeze(fc_series), label='forecast', color='r')
    plt.plot(model_fit.data.endog, label='training')
    plt.plot(df_validation, label='actual')
    plt.title(f'Forecast vs Actual ARIMA params: AR={order[0]}, target_diff={order[1]}, MA={order[1]}')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


def split_time_series_to_train_validation_test(df, split_pct):
    number_of_samples = len(df)
    num_of_observation_in_train = int(np.round(split_pct['train']*number_of_samples, 0))
    num_of_observation_in_validation = int(np.round(split_pct['validation']*number_of_samples, 0))
    '''Create Training and Test'''
    train = pd.DataFrame(df[:(num_of_observation_in_train+1)])
    validation = pd.DataFrame(df[num_of_observation_in_train:
                                       (num_of_observation_in_train+num_of_observation_in_validation)])
    test = pd.DataFrame(df[(num_of_observation_in_train+num_of_observation_in_validation):])
    return train, validation, test


def auto_correlation(selected_diff):
    autocorrelation_plot(selected_diff.dropna())
    plt.title('AR Selection Using Auto-Correlation')
    plt.show()


def main():
    '''
    ARIMA model
    based on: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/ ,
    https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    :return:
    '''
    df = get_the_data(path='https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv')
    split_pct = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
    df, df_validation, df_test = split_time_series_to_train_validation_test(df, split_pct)

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
    '''Partial Auto Correlation Function (PACF)- As the name implies, PACF is a subset of ACF. 
    PACF expresses the correlation between observations made at two points in time while accounting 
    for any influence from other data points. We can use PACF to determine the optimal number of terms to use 
    in the AR model. The number of terms determines the order of the model.'''
    auto_correlation(selected_diff)
    # acf(selected_diff, acf='partial', ylim=(0, 1.2), title='Order of AR Term')
    '''the PACF lag 1 is quite significant since is well above the significance line. 
    Lag 2 turns out to be significant as well, slightly managing to cross the significance limit (blue region). 
    But I am going to be conservative and tentatively fix the p as 1.'''

    '''C. Find the order of the MA term (q)
    An MA term is the error of the lagged forecast.
    The ACF tells how many MA terms are required to remove any auto correlation in the stationary series.
    Auto Correlation Function (ACF)- The correlation between the observations at the current point in time 
    and the observations at all previous points in time. We can use ACF to determine the optimal number of MA terms. 
    The number of terms determines the order of the model.
    If your series is slightly under difference, adding one or more additional AR terms usually makes it up. 
    Likewise, if it is slightly over-difference, try adding an additional MA term.'''

    '''D. After determined the values of p, d and q, fit the ARIMA model.'''
    model_fit = arima_model(df, order=(1, 1, 2))
    print('''\nThe model summary reveals a lot of information. The table in the middle is the coefficients table 
    where the values under ‘coef’ are the weights of the respective terms.
    Notice here the coefficient of the MA2 term is close to zero and the P-Value in ‘P>|z|’ 
    column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.
    So, let’s rebuild the model without the MA2 term.\n''')
    print(model_fit.summary())
    order = (1, 1, 1)
    model_fit = arima_model(df, order=order)
    print('''\nThe model AIC has reduced, which is good. The P Values of the AR1 and MA1 terms have improved 
    and are highly significant (<< 0.05)..\n''')
    print(model_fit.summary())

    '''E. Forecast validation and plot the actual against the fitted values using'''
    actual_vs_predict_plot(model_fit, df_validation, steps=len(df_validation), alpha=0.05, order=order)
    '''From the chart, the ARIMA(1,1,1) model seems to give a directionally correct forecast. 
    But each of the predicted forecasts is consistently below the actual. 
    That means, by adding a small constant to our forecast, the accuracy will certainly improve.'''

    '''F. Manually correction of orders - increase the order of differencing to two, 
    that is set d=2 and iteratively increase p to up to 5 and then q up to 5 to 
    check which model gives least AIC and also look for a chart that gives closer actual and forecasts.
    While doing that the P values of the AR and MA terms should be as close to zero, ideally, less than 0.05.'''
    orders = [(2, 1, 1), (1, 1, 2), (3, 1, 2), (3, 1, 1)]
    for order in orders:
        model_fit = arima_model(df, order=order)
        actual_vs_predict_plot(model_fit, df_validation, steps=len(df_validation), alpha=0.05, order=order)

    '''G. Plot the residuals to ensure there are no patterns (that is, look for constant mean and variance)'''
    residual_plot(model_fit)
    '''The residual errors seem fine with near zero mean and uniform variance.'''
    print(model_fit.summary())


if __name__ == '__main__':
    main()