import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_time_series_data(path_in):
    df = pd.read_csv(f'{path_in}', parse_dates=[0], index_col=[0], dayfirst=True)
    return df


'''explore'''


def explore_data(df):
    print("\ninfo:", df.info())
    print("\ndescribe:\n", df.describe())


def explore_data_value_counts_for_all_features(df):
    print("\nvalue_counts:\n")
    for col in df.columns:
        print(f"\n{col}\n", df[col].value_counts())


''' drop '''


def drop_non_relevant_columns(self, col_list):
    df = self.drop([col_list], axis=1)
    return df


'''visualize'''


def corr(df, feature1, feature2, color="red", kind="reg"):
    '''
    :param df:
    :param feature1:
    :param feature2:
    :param color:
    :param kind: ["hex", "reg"]
    :return:
    '''
    fig = sns.jointplot(x=f'{feature1}', y=f'{feature2}', kind=kind, color=color, data=df)
    plt.show()
    return fig


def bar_count_plot_for_1_feature(df, col, title):
    ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.countplot(f'{col}', data=df)
    plt.title(f"{title}")
    plt.show()
    return plt