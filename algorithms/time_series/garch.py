from random import gauss
from random import seed
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from algorithms.time_series.arima import split_time_series_to_train_validation_test


def background():
    print('''A change in the variance or volatility over time can cause problems when 
    modeling time series with classical methods like ARIMA.
    The ARCH or Autoregressive Conditional Heteroskedasticity method provides a way 
    to model a change in variance in a time series that is time dependent, 
    such as increasing or decreasing volatility. 
    The approach expects the series is stationary, other than the change in variance, 
    meaning it does not have a trend or seasonal component. An ARCH model is used to predict the variance at future time steps.
    Parameters: q- The number of lag squared residual errors to include in the ARCH model.
    In practice, this can be used to model the expected variance on the residuals after another 
    autoregressive model has been used, such as an ARMA or similar.
    
    Generalized Autoregressive Conditional Heteroskedasticity, or GARCH, is an extension of the ARCH model
    that incorporates a moving average component together with the autoregressive component.
    Specifically, the model includes lag variance terms (e.g. the observations if modeling the white noise 
    residual errors of another process), together with lag residual errors from a mean process.
    The introduction of a moving average component allows the model to both model the conditional change in 
    variance over time as well as changes in the time-dependent variance.
    Parameters: p- The number of lag variances to include in the GARCH model.
                q- The number of lag residual errors to include in the GARCH model.
                GARCH(p, q), a GARCH(0, q) is equivalent to an ARCH(q) model.''')


def create_dataset_with_white_noise():
    '''create a simple white noise with increasing variance.
    seed pseudorandom number generator'''
    # seed pseudorandom number generator
    seed(1)
    # create dataset
    data = pd.DataFrame([gauss(0, i * 0.01) for i in range(0, 100)])
    return data


def line_plot(data, title):
    pyplot.plot(data)
    plt.title(title)
    pyplot.show()


def actual_vs_predict_plot(forecast_variance, df_train, df_validation):
    plt.plot(df_train, label='Training')
    '''plot the actual variance'''
    plt.plot(df_validation, label='Actual')
    '''plot forecast variance'''
    plt.plot(forecast_variance, label='Forecast', color='r')
    plt.title('Actual VS Predict')
    plt.legend()
    plt.show()


def main():
    '''
    How to Model Volatility with ARCH and GARCH for Time Series Forecasting
    :source: https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
    '''

    background()

    '''create dataset with white noise'''
    data = create_dataset_with_white_noise()

    '''split into train/test'''
    split_pct = {'train': 0.9, 'validation': 0.1, 'test': 0.0}
    df_train, df_validation, df_test = split_time_series_to_train_validation_test(data, split_pct)

    '''plot'''
    line_plot(df_train, title='Line Plot of Feature with Increasing Variance')

    '''square the dataset to get variance and not sigma- The ACF and PACF plots can then be interpreted 
    to estimate values for p and q, in a similar way as is done for the ARMA model'''
    squared_data = pd.DataFrame([x ** 2 for x in df_train.iloc[:, 0]])
    '''create acf plot'''
    plot_acf(squared_data)
    plt.title('Auto Correlation to Variance')
    plt.show()

    '''define and fit ARCH model, p stands for the number of lag squared residual errors 
    to include in the ARCH model.'''
    model = arch_model(df_train, mean='Zero', vol='ARCH', lags=1, p=6, o=1, q=1)
    model_fit = model.fit()

    '''forecast the test set'''
    y_hat = model_fit.forecast(horizon=len(df_validation), reindex=True)
    new_index = list(range(len(df_train) - 1, len(df_train + df_validation) + 1))
    forecast_variance = pd.DataFrame(index=new_index[1:-1], data=y_hat.variance.values[-1, :])
    actual_vs_predict_plot(forecast_variance, df_train, df_validation)


if __name__ == '__main__':
    main()