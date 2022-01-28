from generic_fun.configuration_param import ArgumentParser


def config_param_path_in():
    r''' create ability to run code with cmd or config parameters, only with one parameter - path_in
    define path to data as parameters configuration-
    for example: --path_in "C:\Users\ronro\Desktop\data"
    or with the cmd, for example: python C:\Users\ronro\PycharmProjects\ML\algorithms\anomaly_and_outliers\anomaly_detection.py --path_in "C:\Users\
    ronro\Desktop\data"
    '''
    parser = ArgumentParser()
    parser.add_argument('--path_in', type=str)
    args = parser.parse_args()
    path_in = args.path_in
    return path_in


def get_data_from_cmd(file_name):
    argv = sys.argv
    path = os.path.join(argv[1], file_name)
    data = pd.read_csv(path)
    return data