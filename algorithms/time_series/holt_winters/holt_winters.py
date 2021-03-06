import os
import sys

import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
# warnings.filterwarnings("error")
# warnings.simplefilter(action='ignore')
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from algorithms.time_series.arima.arima import split_time_series_to_train_validation_test
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.getcwd())
from generic_fun.get_data import config_param_path_in


'''
This code is an implementation of Holt-Winters forecasting
sources: https://medium.com/analytics-vidhya/python-code-on-holt-winters-forecasting-3843808a9873
        https://medium.com/analytics-vidhya/holt-winters-forecasting-13c2e60d983f
'''


def background():
    print('''Holt-Winters Exponential Smoothening (HWES) works on the idea of smoothening the values of a\n
             Univariate Time Series, to use them for forecasting future values.\n
             The idea is to assign exponentially decreasing weights giving more importance to more recent incidents.\n
             So, when we move back in time, we would see diminishing weights.\n\n
             
             Levels-\n
             Looks like stairs.
             
             Trends-\n
             If Levels change in a particular pattern, you can say that the Time Series follows a Trend.\n
             Trends can be linear, square, logarithmic, exponential.\n
             Trend is a vector, as it has both magnitude and direction.\n\n
             
             Seasonality-\n
             Certain patterns that are periodically repeated.
             
             Disadvantages:
             Holt-Winters forecasting cannot handle Time Series data with irregular patterns.
             HWES is a Univariate Forecasting technique and works with Stationary Time Series data.
             ''')


def main(path_in, file_name):
    background()
    '''get the data'''
    airline = pd.read_csv(fr'{path_in}\{file_name}', index_col='Month', parse_dates=True)
    print(airline.shape)

    '''plotting the original data'''
    airline[['Thousands of Passengers']].plot(title='Passenger Data')
    plt.grid()
    plt.show()

    '''decomposing the Time Series
    Look for: Levels, Trends and Seasonality in the data'''
    '''check nan'''
    missing_values_count = airline.isnull().sum(0).sort_values(ascending=False)
    print(missing_values_count)

    '''decompose'''
    decompose_result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')
    decompose_result.plot()
    plt.show()
    '''We can see that the data has all 3 components: Levels, Trends, Seasonality.'''

    '''Define the weight coefficient Alpha and the Time Period, 
    and set the DateTime frequency to a monthly level (m = Time Period).'''
    m = 12
    alpha = 1 / (2 * m)

    '''Fitting the Data with Holt-Winters Exponential Smoothing-
    Fit this data on Single, Double, and Triple Exponential Smoothing respectively.
    While Single HWES can take care of time series data that has levels, 
    Double HWES can as well consider data with trends and Triple HWES can even handle Seasonality.'''

    '''Single Exponential Smoothing'''
    airline['HWES1'] = ExponentialSmoothing(airline['Thousands of Passengers'], freq="MS").fit(smoothing_level=alpha,
                                                                                               optimized=False,
                                                                                               use_brute=True).fittedvalues
    airline[['Thousands of Passengers', 'HWES1']].plot(title='Holt Winters Single Exponential Smoothing')
    plt.grid()
    plt.show()
    '''As expected, it didn???t fit quite well, because single ES doesn't work for data with Trends and Seasonality.'''

    '''Double Exponential Smoothing'''
    airline['HWES2_ADD'] = ExponentialSmoothing(airline['Thousands of Passengers'], trend='add', freq="MS").fit().fittedvalues
    airline['HWES2_MUL'] = ExponentialSmoothing(airline['Thousands of Passengers'], trend='mul', freq="MS").fit().fittedvalues
    airline[['Thousands of Passengers', 'HWES2_ADD', 'HWES2_MUL']].plot(
        title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend')
    plt.grid()
    plt.show()
    '''the fit looks better, but since we know there is Seasonality, we shall move into Triple'''

    '''Triple Exponential Smoothing'''
    airline['HWES3_ADD'] = ExponentialSmoothing(airline['Thousands of Passengers'], trend='add', seasonal='add',
                                                seasonal_periods=12, freq="MS").fit().fittedvalues
    airline['HWES3_MUL'] = ExponentialSmoothing(airline['Thousands of Passengers'], trend='mul', seasonal='mul',
                                                seasonal_periods=12, freq="MS").fit().fittedvalues
    airline[['Thousands of Passengers', 'HWES3_ADD', 'HWES3_MUL']].plot(
        title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality')
    plt.grid()
    plt.show()
    '''looks good!'''

    '''Forecasting with Holt-Winters Exponential Smoothing'''
    '''split the data to train and test'''
    split_pct = {'train': 0.8, 'validation': 0.0, 'test': 0.2}
    train_airline, validation_airline, test_airline = split_time_series_to_train_validation_test(airline, split_pct)

    fitted_model = ExponentialSmoothing(train_airline['Thousands of Passengers'], trend='mul', seasonal='mul',
                                        seasonal_periods=12, freq="MS").fit()
    test_predictions = fitted_model.forecast(len(test_airline))
    '''add the first sample in the prediction to the last in the train so there is no gap in the plot'''
    train_airline_last = pd.DataFrame(test_airline.iloc[0, :])
    frames = [train_airline, train_airline_last.T]
    train_airline_1 = pd.concat(frames)
    train_airline_1['Thousands of Passengers'].plot(legend=True, label='TRAIN')
    test_airline['Thousands of Passengers'].plot(legend=True, label='TEST', figsize=(6, 4))
    test_predictions.plot(legend=True, label='PREDICTION')
    plt.title('Train, Test and Predicted Test using Holt Winters')
    plt.grid()
    plt.show()

    '''plot predict'''
    test_airline[['Thousands of Passengers']].plot(legend=True, label='TEST', figsize=(9, 6))
    test_predictions.plot(title='Test Period', legend=True, label='PREDICTION')
    plt.grid()
    plt.show()

    '''Evaluation'''
    test_data = test_airline['Thousands of Passengers']
    print(f'Mean Absolute Error = {mean_absolute_error(test_data, test_predictions)}')
    print(f'Mean Squared Error = {mean_squared_error(test_data, test_predictions)}')


if __name__ == '__main__':
    file_name = r'international_airline_passengers.csv'
    path_in = config_param_path_in()
    main(path_in, file_name)