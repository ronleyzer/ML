import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import datetime
from datetime import datetime


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


def main():
    pd.options.display.width = 0

    ''' get the data '''
    gym_data = {}
    for key in gym_path.keys():
        gym_data[key] = pd.read_csv(f'{path_in}\\{gym_path[key]}.csv', parse_dates=[0], dayfirst=True)

    '''open output folder'''
    folder = fr"{path_in}\out"
    open_folder(folder)

    number_of_visits_per_costumer = {}
    for key in gym_data.keys():

        '''describe the data'''
        missing_values_count, missing_rows, percent_missing_rows = describe_nan(gym_data[key])

        with open(f'{path_in}\\out\\nan_{key}.txt', 'w') as f:
            f.write(f'\n\n\n*** Describe: {key} ***\n\n'
                    f'Data shape: {gym_data[key].shape}\n\n'
                    f'NAN in dataframe\n\nmissing_values_count:\n'
                    f'{missing_values_count}\n\nmissing_rows: '
                    f'{missing_rows}\n\npercent_missing_rows: {percent_missing_rows}\n\n\n'
                    f'\nDescribe:\n{gym_data[key].describe()}')

        '''deal with missing data - 15-20% of worke locations is missing missing in each gym. 
            Yet it is still possible to use their home location'''

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

        '''create distances between gym-work and gym-home'''
        for place in ['home', 'work']:
            gym_data[key][f'gym_distance_from_{place}'] = \
                gym_data[key][[f'user_{place}_lat', f'user_{place}_long', 'visit_lat', 'visit_long'
                               ]].apply(lambda x: try_distance(x[:2], x[2:]), axis=1)
        '''create dummy for what is closer'''
        gym_data[key]['home_closer_than_work'] = np.where(gym_data[key]['gym_distance_from_home'] <=
                                                          gym_data[key]['gym_distance_from_work'], True, False)
        '''how many time a customer visit one time and don't come back'''
        # number_of_visits_per_costumer[key] = gym_data[key].value_counts(['device_id'])
        gym_data[key]['number_of_visit'] = gym_data[key]['device_id'].apply(lambda x: gym_data[key][gym_data[key]['device_id'] == x].shape[0])
        gym_data[key] = gym_data[key].sort_values(by='device_id')
        gym_data[key]['first_for_device_id'] = gym_data[key]['device_id'] != gym_data[key]['device_id'].shift()
        number_of_visits_per_costumer[key] = gym_data[key][gym_data[key]['first_for_device_id']]

    '''visualize'''
    print('visualize')
    param_list = ['gym_distance_from_home', 'gym_distance_from_work', 'month', 'duration', 'start_hour']
    for weight in ['customer_weight', 'visit_weight', 'customer_and_visit_weight']:
        '''open folders to each weight option'''
        folder = fr'{path_in}\out\{weight}'
        open_folder(folder)

        '''cumulative quantiles'''
        for param in param_list:
            for key in gym_data.keys():
                gym_data[key][param+'_rank'] = gym_data[key][param].rank(pct=True)
                gym_data[key].sort_values(by=f'{param}_rank', inplace=True)
                w_mean = np.average(gym_data[key][param], weights=gym_data[key][weight])
                plt.plot(gym_data[key][f'{param}_rank'], gym_data[key][f'{param}'], label=f'{key}, w_mean: {np.round(w_mean,1)}')
            plt.legend()
            plt.title(f"{param} weight: {weight}")
            plt.savefig(fr"{folder}\{weight}_cumulative_distribution_{param}.png")
            plt.close()

        '''histograms'''
        for param in param_list:
            for key in gym_data.keys():
                df = gym_data[key].copy()
                if param == 'gym_distance_from_work':
                    df = df.dropna()
                df = df[df[f'{param}_rank'] < 0.94]
                w_mean = np.average(df[param], weights=df[weight])
                plt.hist(df[param], label=f'{key}, w_mean: {np.round(w_mean,1)}', histtype="step", density=True, bins=50, weights=df[weight])
            plt.legend()
            plt.title(f"{param} weight: {weight}")
            plt.savefig(fr"{folder}\{weight}_hist_{param}.png")
            plt.close()

        for key in number_of_visits_per_costumer.keys():
            w_mean = np.average(number_of_visits_per_costumer[key][f'number_of_visit'],
                                      weights=number_of_visits_per_costumer[key][weight])
            plt.hist(number_of_visits_per_costumer[key]['number_of_visit'], label=f'{key}, w_mean: {np.round(w_mean,1)}', histtype="step", density=True,
                     bins=50, weights=number_of_visits_per_costumer[key][f'{weight}'])
        plt.legend()
        plt.title(f"number of visits per costumer weight: {weight}")
        plt.savefig(fr"{folder}\{weight}_hist_gym_number_of_visits_per_costumer_{param}.png")
        plt.close()

        '''save data with after feature_engineering'''
        for key in gym_data.keys():
            gym_data[key].to_csv(fr'{path_in}\{key}.csv')


if __name__ == '__main__':
    path_in = r'C:\Users\ron.l\Desktop\pla'
    gym_path = {
                '299': 'visits_Planet_Fitness_299_Molly_Lane_Woodstock_GA_United_States_2018-02-01_2019-02-01',
                '1570': 'visits_Planet_Fitness_1570_Holcomb_Bridge_Road_Roswell_Georgia_United_States_2018-02-01_2019-02-01',
                '10790': 'visits_Planet_Fitness_10790_Alpharetta_Hwy_Roswell_GA_30076_Roswell_GA_United_States_2018-02-01_2019-02-01',
                '13071': 'visits_Planet_Fitness_13071_Highway_9_Milton_GA_United_States_2018-02-01_2019-02-01',
    }
    main()