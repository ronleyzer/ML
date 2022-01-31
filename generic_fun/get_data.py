from generic_fun.configuration_param import ArgumentParser


def config_param_path_in():
    r'''Reads the parameters from the cmd or the config, and returns the relevant path
        for example: --path_in "C:\Users\ronro\Desktop\data" '''
    parser = ArgumentParser()
    parser.add_argument('--path_in', type=str)
    args = parser.parse_args()
    path_in = args.path_in
    return path_in
