import pandas as pd
import os
import seaborn as sns
import numpy as np


def open_folder(folder):
    '''open folder'''
    if not os.path.exists(folder):
        os.makedirs(folder)


def describe_nan(df):
    missing_values_count = df.isnull().sum(0).sort_values(ascending=False)
    missing_rows = df.shape[0] - df.dropna().shape[0]
    percent_missing_rows = missing_rows / df.shape[0]
    return missing_values_count, missing_rows, percent_missing_rows


def main():
    pd.options.display.width = 0

    ''' get the data '''
    gym_data = {}
    for key in gym_path.keys():
        gym_data[key] = pd.read_csv(f'{path_in}\\{gym_path[key]}.csv', index_col=[0], parse_dates=[0], dayfirst=True)

    '''open output folder'''
    folder = fr"{path_in}\out"
    open_folder(folder)

    for key in gym_data.keys():
        df = gym_data[key]

        '''describe the data'''
        missing_values_count, missing_rows, percent_missing_rows = describe_nan(df)

        with open(f'{path_in}\\out\\nan_{key}.txt', 'w') as f:
            f.write(f'\n\n\n*** Describe: {key} ***\n\n'
                    f'Data shape: {df.shape}\n\n'
                    f'NAN in dataframe\n\nmissing_values_count:\n'
                    f'{missing_values_count}\n\nmissing_rows: '
                    f'{missing_rows}\n\npercent_missing_rows: {percent_missing_rows}\n\n\n'
                    f'\nDescribe:\n{df.describe()}')

        '''deal with missing data - users work have missing data in all gym dfs. 
            Yet it is still possible yo use their home location'''

        '''feature engendering'''
        '''create distances between gym-work and gym-home'''
        for place in ['home', 'work']:
            a = np.array((df[f'user_{place}_lat'], df[f'user_{place}_long']))
            b = np.array((df['visit_lat'], df['visit_long']))
            df[f'gym_distance_from_{place}'] = np.linalg.norm(a-b)

        df['home_closer_than_work'] = np.where(df['gym_distance_from_home'] <= df['gym_distance_from_work'], True, False)




if __name__ == '__main__':
    path_in = r'C:\Users\ron.l\Desktop\pla'
    gym_path = {
                '299': 'visits_Planet_Fitness_299_Molly_Lane_Woodstock_GA_United_States_2018-02-01_2019-02-01',
                '1570': 'visits_Planet_Fitness_1570_Holcomb_Bridge_Road_Roswell_Georgia_United_States_2018-02-01_2019-02-01',
                '10790': 'visits_Planet_Fitness_10790_Alpharetta_Hwy_Roswell_GA_30076_Roswell_GA_United_States_2018-02-01_2019-02-01',
                '13071': 'visits_Planet_Fitness_13071_Highway_9_Milton_GA_United_States_2018-02-01_2019-02-01',
    }
    main()