import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from algorithms.deal_with_new_data import describe_nan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from featurewiz import featurewiz
from sklearn.datasets import make_regression


''' get the data '''
path_in = r'C:\Users\ron.l\Desktop\ML\Kaggel\house-prices-advanced-regression-techniques'
df = pd.read_csv(f'{path_in}\\train.csv', index_col=[0], parse_dates=[0], dayfirst=True)


''''clean the data'''
'''describe nan'''
missing_values_count, missing_rows, percent_missing_rows = describe_nan(df)
with open(f'{path_in}\\out\\nan.txt', 'w') as f:
    f.write(f'NAN in dataframe\n\nmissing_values_count:\n {missing_values_count}\n\nmissing_rows: '
            f'{missing_rows}\n\npercent_missing_rows: {percent_missing_rows}\n')

'''column nan'''
column_nan_drop_li = ['Fence', 'Alley', 'MiscFeature', 'PoolQC', 'FireplaceQu']
for col in column_nan_drop_li:
    del df[col]

'''row nan - drop if more than 0 values are missing in a row'''
df = df.dropna()
# print("after dropping")
# describe_nan(df)

'''y nan'''
print("y_nan:", df['SalePrice'].isnull().sum(0))
df = df.dropna(subset=['SalePrice'])

'''value count for categorical features'''
for col in df.loc[:, df.dtypes == np.object]:
    col_count = df[col].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(col_count.index, col_count.values, alpha=0.9)
    plt.title(f"{col} Value Count")
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(f"{col}", fontsize=12)
    plt.savefig(f"{path_in}\\out\\value_count\\{col}.png")
    plt.close()

'''drop categorical less relevant or not adding information'''
keep_cat_li = ['CentralAir',
               'Exterior1st',
               # 'Neighborhood'
               ]
drop_cat_li = list(set(df.loc[:, df.dtypes == np.object].columns) - set(keep_cat_li))
df = df.drop(drop_cat_li, axis=1)

'''change categorical feature to int or gathering and transformed to dummies'''
replace_map = {'CentralAir': {'N': '0', 'Y': '1'},
               'Exterior1st': {'Stone': 'Other', 'Wd Sdng': 'wood', 'BrkComm': 'Brick', 'ImStucc': 'Other',
                               'Plywood': 'wood', 'AsbShng': 'Other', 'Stucco': 'Other', 'CBlock': 'Other',
                               'WdShing': 'wood', 'BrkFace': 'Brick'},
               # 'Neighborhood': {'Blueste', 'BrkSide', 'CollgCr', 'NAmes', 'ClearCr', 'StoneBr', 'NoRidge', 'IDOTRR',
               #                  'Veenker', 'Gilbert', 'NPkVill', 'MeadowV', 'NWAmes', 'OldTown', 'Edwards', 'BrDale',
               #                  'Crawfor', 'SawyerW', 'Timber', 'NridgHt', 'Blmngtn', 'Somerst', 'Sawyer', 'SWISU', 'Mitchel'}
               }
df = df.replace(replace_map)
df['CentralAir'] = df['CentralAir'].astype('float64')
df = pd.get_dummies(df, prefix=['Exterior1st'])

'''histogram for int and flot features'''
for col in df.loc[:, df.dtypes != np.object]:
    sns.distplot(df[col])
    plt.savefig(f"{path_in}\\out\\hist\\{col}.png")
    plt.close()

'''transformed object to '''
'''split to train, dev(validation) and test '''
X = df.iloc[:, :-1]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


''' *** feature selection *** '''


''' *** model options *** '''
models_acc = {}

'''linear'''
'''ridge regularization - low alpha is more generalization'''
models_acc['ridge'] = {}
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
train_accuracy = ridge_model.score(X_train, y_train)
val_accuracy = ridge_model.score(X_val, y_val)
models_acc['ridge']['train'] = train_accuracy
models_acc['ridge']['val'] = val_accuracy

'''lasso regularization - difference of lasso and ridge regression is that some of the coefficients can be zero'''
models_acc['lasso'] = {}
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
train_accuracy = lasso_model.score(X_train, y_train)
val_accuracy = lasso_model.score(X_val, y_val)
models_acc['lasso']['train'] = train_accuracy
models_acc['lasso']['val'] = val_accuracy

'''trees'''
models_acc['rf_reg'] = {}
rf_reg = RandomForestRegressor(max_depth=2, random_state=1)
rf_reg.fit(X_train, y_train)
train_accuracy = rf_reg.score(X_train, y_train)
val_accuracy = rf_reg.score(X_val, y_val)
models_acc['rf_reg']['train'] = train_accuracy
models_acc['rf_reg']['val'] = val_accuracy

