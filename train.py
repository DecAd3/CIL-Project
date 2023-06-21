import sys
import yaml
import argparse
from utils import _read_df_in_format, _convert_df_to_matrix, preprocess


def process_config(path):
    with open(path, 'r') as file:
        yaml_str = file.read()

    data = yaml.safe_load(yaml_str)
    args = argparse.Namespace()

    training_args = data['args']['training_args']
    args.train_data = training_args['train_data']
    args.test_save_dir = training_args['test_save_dir']
    args.model_load_path = training_args['model_load_path']
    args.model_save_path = training_args['model_save_path']
    args.random_seed = training_args['random_seed']
    args.device = training_args['device']
    args.normalization = training_args['normalization']
    args.imputation = training_args['imputation']

    experiment_args = data['args']['experiment_args']
    args.model_name = experiment_args['model_name']
    # others

    return args


def train(args):
    df = _read_df_in_format(args.train_data)
    data, is_provided = _convert_df_to_matrix(df, 10000, 1000)
    data = preprocess(data, 10000, 1000)


if __name__ == '__main__':
    args = process_config(sys.argv[1])
    train(args)
