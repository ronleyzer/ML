import sys
import pandas as pd
from scipy.io import loadmat, wavfile


def get_data_from_cmd(file_name: str, read_option: str = 'csv_time_series'):
    r'''
    this function take a string writen in cmd that includes the path to the code we like to run
    and the path to the local folder with the data, for example:
    python ML\algorithms\dimensionality_reduction\pca.py C:\Users\ronro\Desktop\data\
    :param file_name: file name or path, for example: \anomaly_detection\Melbourne_housing_FULL.csv
    :param read_option: csv as default
    :return: the data
    '''
    argv = sys.argv
    path = argv[1] + file_name
    if read_option == 'csv_time_series':
        data = pd.read_csv(path, index_col=[0], parse_dates=[0], dayfirst=True)
    if read_option == 'csv':
        data = pd.read_csv(path)
    if read_option == 'mat':
        data = loadmat(path)
    if read_option == 'wav':
        data = wavfile.read(path)
    return data