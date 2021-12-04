import matplotlib.pyplot as plt
import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str


def background():
    print('The VAR model stands for Vector Auto Regression. We use VAR when we want to predict\n'
          'two or more time series that influence each other. For example in Macrobot we have a\n '
          'high related assets we want to predict (stocks), so VAR could help us with that.\n')


def get_the_data():
    mdata = sm.datasets.macrodata.load_pandas().data
    return mdata


def prepare_the_dates_index(mdata):
    dates = mdata[['year', 'quarter']].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]
    quarterly = dates_from_str(quarterly)
    mdata = mdata[['realgdp', 'realcons', 'realinv']]
    mdata.index = pandas.DatetimeIndex(quarterly)
    data = np.log(mdata).diff().dropna()
    return data


def main():
    '''
    :source: https://www.statsmodels.org/stable/vector_ar.html#
             https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
             https://www.scribbr.com/statistics/akaike-information-criterion/
    '''
    background()
    mdata = get_the_data()
    data = prepare_the_dates_index(mdata)

    '''make a VAR model'''
    model = VAR(data, freq='q')

    '''Lag order selection - pick the order that gives a model with least AIC, in this case its 3.
    The VAR model requires a lag parameter. The selection of the parameter is decides using the lowest AIC (Akaike Information Criterion) for different lag options.
    The AIC is a mathematical method for evaluating how well a model fits the data it was generated from. AIC is calculated from:
    •	the number of independent variables used to build the model.
    •	the maximum likelihood estimate of the model (how well the model reproduces the data).
    The best-fit model according to AIC is the one that explains the greatest amount of variation using the fewest possible independent variables.
    '''
    x = model.select_order(maxlags=12)
    print(x.summary())

    '''estimation: fit method with the desired lag order '''
    results = model.fit(3)
    print(results.summary())

    '''visualize the data'''
    results.plot()
    plt.show()
    '''auto correlation function'''
    results.plot_acorr()
    plt.show()

    '''Forecasting'''
    lag_order = results.k_ar
    results.forecast(data.values[-lag_order:], 5)
    results.plot_forecast(10, plot_stderr=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()