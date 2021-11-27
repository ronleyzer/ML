import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import datetime
from datetime import datetime
import matplotlib as mpl
import matplotlib.font_manager


def open_folder(folder):
    '''open folder'''
    if not os.path.exists(folder):
        os.makedirs(folder)


def describe_nan(df):
    '''
    :param df:
    :return: missing_values_count, missing_rows, percent_missing_rows
    '''
    missing_values_count = df.isnull().sum(0).sort_values(ascending=False)
    missing_rows = df.shape[0] - df.dropna().shape[0]
    percent_missing_rows = missing_rows / df.shape[0]
    return missing_values_count, missing_rows, percent_missing_rows


def try_distance(x, y):
    '''
    :param x: array of coordinates of point 1
    :param y: array of coordinates of point 2
    :return: returns the euclidean distance if arrays x and y are no empty
    '''
    try:
        return distance.euclidean(x, y)
    except:
        return np.nan


def get_max_height(bars):
    max_height = 0
    min_height = 0
    for bar in bars:
        if bar.get_height()>max_height:
            max_height = bar.get_height()
        if bar.get_height()<min_height:
            min_height = bar.get_height()
    return max_height, min_height


def add_text_to_bar(bars, ax):
    max_height, min_height = get_max_height(bars)
    for bar in bars:
        if bar.get_height() > 0:
            additional_pos = max_height * 0.01
        else:
            additional_pos = min_height * 0.2
        bar_color = bar.get_facecolor()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + additional_pos,
            str(round(bar.get_height(), 1)) + '%',
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )


def bar_plot(df, feature, title):
    # Use Matplotlib's font manager to rebuild the font library.
    mpl.font_manager._rebuild()
    # Use the newly integrated Roboto font family for all text.
    plt.rc('font', family='Liberation Sans')
    fig, ax = plt.subplots()
    plt.figure(figsize=(24, 12))
    # Save the chart so we can loop through the bars below.
    bars = ax.bar(
        x=np.arange(df[feature].size),
        height=df[feature],
        tick_label=[name[:3] for name in list(df.index)]
    )

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    # Add text annotations to the top of the bars.
    add_text_to_bar(bars, ax)
    # Add labels and a title.
    ax.set_xlabel(' ', labelpad=15, color='#333333')
    ax.set_ylabel(' ', labelpad=15, color='#333333')
    ax.set_title(title, pad=15, color='#333333',
                 weight='bold')
    fig.tight_layout()


def bar_plot_multiple(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, title=''):
    # Check if colors where provided, otherwise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Number of bars per group
    n_bars = len(data)
    # The width of a single bar
    bar_width = total_width / n_bars
    # List containing handles for the drawn bars, used for the legend
    bars = []
    group_centers = []
    max_y = 0
    min_y = 0
    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        # group_centers.append(x_offset)
        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
            if np.max(y) > max_y:
                max_y = np.max(y)
            if np.min(y) < min_y:
                min_y = np.min(y)
            if i == 2:
                group_centers.append(x + x_offset)
            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])
    # fix x axis
    ax.set_xticks(group_centers)
    ax.set_xticklabels(list(data.index))
    # text
    plt.ylim(1.3 * min_y, 1.3*max_y)
    if legend:
        ax.legend(bars, data.keys())
        leg = ax.get_legend()
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i].set_color(colors[i % len(colors)])
    add_text_to_bar(bars, ax)
    plt.title(title, pad=15, color='#333333', weight='bold')


def two_sides_bar_plot(upper_plot_y, lowe_plot_y, x, title):
    plt.figure(figsize=(16, 8))
    y1 = upper_plot_y
    y2 = lowe_plot_y
    plt.bar(x, +y1, facecolor='#C0CA33', edgecolor='white')  # specify edgecolor by name
    plt.bar(x, y2, facecolor='#FF9800', edgecolor='white')
    for val, y in zip(x, y1):
        plt.text(val, y + 0.05, str('%.1f' % y)+'%', ha='center', va='bottom', fontsize=10)
    for val, y in zip(x, y2):
        plt.text(val, y - 4.8, str('%.1f' % y)+'%', ha='center', va='bottom', fontsize=10)

    plt.xlim(0, 12)
    plt.ylim(-np.max(y1)-7, np.max(y1)+10)
    plt.title(title, pad=15, color='#333333', weight='bold')
    plt.tight_layout()


def create_weighted_df(df, weights_column):
    weighted_df = df.copy()
    weighted_df['rounded_weight'] = np.round(weighted_df[weights_column]).astype('int64')
    weighted_df = weighted_df.loc[weighted_df.index.repeat(weighted_df['rounded_weight'])]
    return weighted_df


def main():

    pd.options.display.width = 0

    ''' get the data '''
    gym_data = {}
    gym_data_weighted = {}
    for key in gym_path.keys():
        gym_data[key] = pd.read_csv(f'{path_in}\\{gym_path[key]}.csv', parse_dates=[0], dayfirst=True)
        gym_data[key]['sum_weight'] = gym_data[key]['visit_weight'] + gym_data[key]['customer_weight']
        gym_data_weighted[key] = {}
        for weight in ['visit_weight', 'customer_weight', 'sum_weight']:
            gym_data_weighted[key][weight] = create_weighted_df(gym_data[key], weight)

    '''open output folder'''
    folder = fr"{path_in}\out"
    open_folder(folder)

    number_of_visits_per_costumer = {}
    monthly_df_dic = {}
    for key in gym_data.keys():

        '''describe the data'''
        # missing_values_count, missing_rows, percent_missing_rows = describe_nan(gym_data[key])
        #
        # with open(f'{path_in}\\out\\nan_{key}.txt', 'w') as f:
        #     f.write(f'\n\n\n*** Describe: {key} ***\n\n'
        #             f'Data shape: {gym_data[key].shape}\n\n'
        #             f'NAN in dataframe\n\nmissing_values_count:\n'
        #             f'{missing_values_count}\n\nmissing_rows: '
        #             f'{missing_rows}\n\npercent_missing_rows: {percent_missing_rows}\n\n\n'
        #             f'\nDescribe:\n{gym_data[key].describe()}')
        #     '''15-20% of work locations is missing missing in each gym'''

        '''feature engineering'''
        print('feature engineering')

        '''sum of customer_weight and visit_weight'''
        gym_data[key]['customer_and_visit_weight'] = sum(gym_data[key]['customer_weight'], gym_data[key]['visit_weight'])

        '''split date_time'''
        gym_data[key]['start_time'] = gym_data[key]['visit_start_time'].apply(lambda x: x.split(" ")[1])
        gym_data[key]['start_hour'] = gym_data[key]['start_time'].apply(lambda x: x.split(":")[0]).astype('int64')
        gym_data[key]['end_time'] = gym_data[key]['visit_end_time'].apply(lambda x: x.split(" ")[1])
        gym_data[key]['end_hour'] = gym_data[key]['end_time'].apply(lambda x: x.split(":")[0]).astype('int64')
        gym_data[key]['visit_start_time_td'] = gym_data[key]['visit_start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        gym_data[key]['visit_end_time_td'] = gym_data[key]['visit_end_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        gym_data[key]['duration'] = gym_data[key][['visit_end_time_td', 'visit_start_time_td']].apply(lambda x: divmod((x[0] - x[1]).total_seconds(), 60)[0], axis=1)
        gym_data[key]['date'] = gym_data[key]['visit_start_time'].apply(lambda x: x.split(" ")[0])
        gym_data[key]['month'] = gym_data[key]['date'].apply(lambda x: x.split("-")[1]).astype('int64')
        gym_data[key]['date'] = gym_data[key]['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        gym_data[key]['weekday'] = gym_data[key]['date'].apply(lambda x: x.weekday())

        '''create indicator for first or last time for each customer'''
        gym_data[key]['visit_start_time'] = gym_data[key]['visit_start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        '''first workout'''
        gym_data[key]['first_workout_date'] = gym_data[key]['device_id']\
            .apply(lambda x: (gym_data[key][gym_data[key]['device_id'] == x]['visit_start_time']).min())
        gym_data[key]['is_first_workout_date'] = gym_data[key]['first_workout_date'] == gym_data[key]['visit_start_time']
        '''last workout'''
        gym_data[key]['last_workout_date'] = gym_data[key]['device_id']\
            .apply(lambda x: (gym_data[key][gym_data[key]['device_id'] == x]['visit_end_time']).max())
        gym_data[key]['is_last_workout_date'] = gym_data[key]['last_workout_date'] == gym_data[key]['visit_end_time']

        '''over_time'''
        monthly_df = pd.DataFrame()
        estimate_monthly_df = pd.DataFrame()
        for feature in ['is_first_workout_date', 'is_last_workout_date']:
            monthly_df[feature] = gym_data[key].groupby(gym_data[key]['visit_start_time'].dt.strftime('%B'))[
                feature].sum()

        for feature in ['visit_id', 'device_id']:
            monthly_df[feature] = gym_data[key].groupby(gym_data[key]['visit_start_time'].dt.strftime('%B'))[
                feature].nunique()

        '''sort'''
        month_to_int = {'April': 4, 'August': 8, 'December': 12, 'February': 2, 'January': 13, 'July': 7, 'June': 6,
                        'March': 3, 'May': 5, 'November': 11, 'October': 10, 'September': 9}
        monthly_df['int_month'] = monthly_df.index
        monthly_df['int_month'] = monthly_df['int_month'].apply(lambda x: month_to_int[x])
        monthly_df = monthly_df.sort_values(by='int_month')

        '''create shifts and pct change'''
        for feature in ['visit_id', 'device_id']:
            monthly_df[f'{feature}_shift'] = monthly_df[feature].shift(1)
            monthly_df[f'{feature}_pct_cng'] = np.round((monthly_df[feature].pct_change() * 100), 1)
        monthly_df[f'pct_new_comers'] = np.round(((monthly_df['is_first_workout_date']/monthly_df['device_id_shift']) * 100), 1)
        monthly_df[f'pct_leavers'] = np.round(((monthly_df['is_last_workout_date'].shift(1)/monthly_df['device_id_shift']) * -100), 1)

        '''save monthly_df'''
        monthly_df_dic[key] = monthly_df

        '''create distances between gym-work and gym-home'''
        # for place in ['home', 'work']:
        #     gym_data[key][f'gym_distance_from_{place}'] = \
        #         gym_data[key][[f'user_{place}_lat', f'user_{place}_long', 'visit_lat', 'visit_long'
        #                        ]].apply(lambda x: try_distance(x[:2], x[2:]), axis=1)
        # '''create dummy for what is closer'''
        # gym_data[key]['home_closer_than_work'] = np.where(gym_data[key]['gym_distance_from_home'] <=
        #                                                   gym_data[key]['gym_distance_from_work'], True, False)
        # '''how many time a customer visit one time and don't come back'''
        # # number_of_visits_per_costumer[key] = gym_data[key].value_counts(['device_id'])
        # gym_data[key]['number_of_visit'] = gym_data[key]['device_id'].apply(lambda x: gym_data[key][gym_data[key]['device_id'] == x].shape[0])
        # gym_data[key] = gym_data[key].sort_values(by='device_id')
        # gym_data[key]['first_for_device_id'] = gym_data[key]['device_id'] != gym_data[key]['device_id'].shift()
        # number_of_visits_per_costumer[key] = gym_data[key][gym_data[key]['first_for_device_id']]

        '''visualize'''
        print('visualize')

        'features histogram'
        # gym_data[key][gym_data[key].columns.tolist()].hist(density=True, figsize=(15, 10))
        # plt.tight_layout()
        # plt.show()

    '''time series'''
    '''visit pct change for gym 10790'''
    bar_plot(monthly_df_dic['10790'], 'visit_id_pct_cng', title='Alpharetta Monthly Visits Change MOM')
    # plt.savefig(fr"{path_in}\Alpharetta Monthly Visits Change MOM.png")
    plt.show()
    plt.close()
    '''visit pct change for all gyms '''
    monthly_df_visits = pd.DataFrame()

    for key in monthly_df_dic.keys():
        monthly_df_visits[key] = monthly_df_dic[key]['visit_id_pct_cng']
    monthly_df_visits.rename(columns={'10790':'Alpharetta', '299':'Molly Lane', '1570':'Holcomb', '13071':'Highway 9'}, inplace=True)

    month_list = [['March', 'April', 'May', 'June'], ['July', 'August', 'September', 'October'], ['November', 'December', 'January']]
    for months in month_list:
        fig, ax = plt.subplots()
        bar_plot_multiple(ax, monthly_df_visits.T[months], total_width=.8, single_width=.9,
                          title='Planet Fitness Branches Visits Change MOM')
        # plt.savefig(fr"{path_in}\Planet Fitness Branches Visits Change MOM {months}.png")
        plt.show()
        plt.close()
    '''two sides leavers and comers'''
    two_sides_bar_plot(upper_plot_y=monthly_df_dic['10790']['pct_new_comers'],
                       lowe_plot_y=monthly_df_dic['10790']['pct_leavers'],
                       x=monthly_df_dic['10790'].index, title='Alpharetta Monthly New Comers and Leavers Change MOM')
    # plt.savefig(fr"{path_in}\Alpharetta Monthly New Comers and Leavers Change MOM.png")
    plt.show()
    plt.close()

    #
    # '''histograms'''
    # param_list = ['gym_distance_from_home', 'gym_distance_from_work', 'month', 'duration', 'start_hour']
    # for weight in ['customer_weight', 'visit_weight', 'customer_and_visit_weight']:
    #     '''open folders to each weight option'''
    #     folder = fr'{path_in}\out\{weight}'
    #     open_folder(folder)
    #
    #     '''cumulative quantiles'''
    #     for param in param_list:
    #         for key in gym_data.keys():
    #             gym_data[key][param+'_rank'] = gym_data[key][param].rank(pct=True)
    #             gym_data[key].sort_values(by=f'{param}_rank', inplace=True)
    #             w_mean = np.average(gym_data[key][param], weights=gym_data[key][weight])
    #             plt.plot(gym_data[key][f'{param}_rank'], gym_data[key][f'{param}'], label=f'{key}, w_mean: {np.round(w_mean,1)}')
    #         plt.legend()
    #         plt.title(f"{param} weight: {weight}")
    #         plt.savefig(fr"{folder}\{weight}_cumulative_distribution_{param}.png")
    #         plt.close()
    #
    #     '''histograms'''
    #     for param in param_list:
    #         for key in gym_data.keys():
    #             df = gym_data[key].copy()
    #             if param == 'gym_distance_from_work':
    #                 df = df.dropna()
    #             df = df[df[f'{param}_rank'] < 0.94]
    #             w_mean = np.average(df[param], weights=df[weight])
    #             plt.hist(df[param], label=f'{key}, w_mean: {np.round(w_mean,1)}', histtype="step", density=True, bins=50, weights=df[weight])
    #         plt.legend()
    #         plt.title(f"{param} weight: {weight}")
    #         plt.savefig(fr"{folder}\{weight}_hist_{param}.png")
    #         plt.close()
    #
    #     for key in number_of_visits_per_costumer.keys():
    #         w_mean = np.average(number_of_visits_per_costumer[key][f'number_of_visit'],
    #                                   weights=number_of_visits_per_costumer[key][weight])
    #         plt.hist(number_of_visits_per_costumer[key]['number_of_visit'], label=f'{key}, w_mean: {np.round(w_mean,1)}', histtype="step", density=True,
    #                  bins=50, weights=number_of_visits_per_costumer[key][f'{weight}'])
    #     plt.legend()
    #     plt.title(f"number of visits per costumer weight: {weight}")
    #     plt.savefig(fr"{folder}\{weight}_hist_gym_number_of_visits_per_costumer_{param}.png")
    #     plt.close()
    #
    #     '''save data with after feature_engineering'''
    #     for key in gym_data.keys():
    #         gym_data[key].to_csv(fr'{path_in}\{key}.csv')


if __name__ == '__main__':
    path_in = r'C:\Users\ron.l\Desktop\pla'
    gym_path = {
                '10790': 'visits_Planet_Fitness_10790_Alpharetta_Hwy_Roswell_GA_30076_Roswell_GA_United_States_2018-02-01_2019-02-01',
                '299': 'visits_Planet_Fitness_299_Molly_Lane_Woodstock_GA_United_States_2018-02-01_2019-02-01',
                '1570': 'visits_Planet_Fitness_1570_Holcomb_Bridge_Road_Roswell_Georgia_United_States_2018-02-01_2019-02-01',
                '13071': 'visits_Planet_Fitness_13071_Highway_9_Milton_GA_United_States_2018-02-01_2019-02-01',
    }
    main()