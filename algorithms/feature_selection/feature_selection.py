from sklearn.datasets import load_boston
import pandas as pd
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler


def get_the_data():
    boston = load_boston()
    print(boston.data.shape)  # for dataset dimension
    print(boston.feature_names)  # for feature names
    # print(boston.target)  # for target variable
    print(boston.DESCR)  # for data description
    bos = pd.DataFrame(boston.data, columns=boston.feature_names)
    bos['Price'] = boston.target
    X = bos.drop("Price", axis=1)  # feature matrix
    y = bos['Price']  # target feature
    return X, y


def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype='float64')
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            # model = sm.OLS(target, sm.add_trend(data[best_features+[new_column]], trend="c")).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if (min_p_value < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def backward_elimination(data, target, significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break
        else:
            break
    return best_features


def main():
    ''' feature selection using Forward, Backward and Bi-directional selection
    source: https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/'''
    X, y = get_the_data()

    '''***** Wrapper methods *****'''
    ''' 1. Forward selection-
    A. Choose a significance level (e.g. SL = 0.05 with a 95% confidence).
    B. Fit all possible simple regression models by considering one feature at a time.
       Total ’n’ models are possible. Select the feature with the lowest p-value.
    C. Fit all possible models with one extra feature added to the previously selected feature(s).
    D. Again, select the feature with a minimum p-value. if p_value < significance level then go to Step 3, 
    otherwise terminate the process.
    '''
    best_features_forward_selection = forward_selection(X, y, significance_level=0.05)
    '''Forward selection using built-in functions- Sequential Forward Selection(sfs)'''
    sfs = SFS(LinearRegression(),
              k_features=8,
              forward=True,
              floating=False,
              scoring='r2',
              cv=0)
    sfs.fit(X, y)
    best_features_forward_selection = sfs.k_feature_names_  # to get the final set of features

    '''2. Backward elimination-
    A. Choose a significance level (e.g. SL = 0.05 with a 95% confidence).
    B. Fit a full model including all the features.
    C. Consider the feature with the highest p-value. If the p-value > significance level then go to Step 4, 
       otherwise terminate the process.
    D. Remove the feature which is under consideration.
    E. Fit a model without this feature. Repeat the entire process from Step 3. 
    '''
    best_features_backward_elimination = backward_elimination(X, y, significance_level=0.05)
    '''Backward elimination using built-in functions- Sequential Forward Selection(sfs)'''
    sbs = SFS(LinearRegression(),
              k_features=8,
              forward=False,
              floating=False,
              cv=0)
    sbs.fit(X, y)
    best_features_backward_elimination = sbs.k_feature_names_

    '''Additional Note
    Here we are directly using the optimal value of k_features argument in both forward selection and 
    backward elimination.     In order to find out the optimal number of significant features, 
    we can use the hit and trial method for different values of k_features and make the final decision 
    by plotting it against the model performance.'''
    sfs1 = SFS(LinearRegression(),
               k_features=(3, 13),
               forward=True,
               floating=False,
               cv=0)
    sfs1.fit(X, y)
    fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
    plt.title('Sequential Forward Selection (w. StdErr)')
    plt.grid()
    plt.show()

    '''3. Bi-directional elimination-
    A. Choose a significance level to enter and exit the model (e.g. SL_in = 0.05 and SL_out = 0.05 with 95% confidence).
    B. Perform the next step of forward selection (newly added feature must have p-value < SL_in to enter).
    C. Perform all steps of backward elimination (any previously added feature with p-value>SL_out is ready to exit the model).
    D. Repeat steps 2 and 3 until we get a final optimal set of features.
    '''
    best_features_stepwise_selection = stepwise_selection(X, y)
    # Sequential Forward Floating Selection(sffs)
    sffs = SFS(LinearRegression(),
               k_features=(1, 8),
               forward=True,
               floating=True,
               cv=0)
    sffs.fit(X, y)
    best_features_stepwise_selection = sffs.k_feature_names_

    '''compare feature selection methods'''

    selected_features = pd.concat([pd.Series(best_features_forward_selection),
                                   pd.Series(best_features_backward_elimination),
                                   pd.Series(best_features_stepwise_selection)], axis=1)
    selected_features = selected_features.rename(columns={0: 'Forward', 1:'Backward', 2:'Forward & Backward'})


if __name__ == '__main__':
    main()