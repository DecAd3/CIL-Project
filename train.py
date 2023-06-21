import sys
import yaml
import argparse
from utils import load_data, mean_impute


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

    experiment_args = data['args']['experiment_args']
    args.model_name = experiment_args['model_name']
    # others

    return args


def train(args):
    data = load_data(args.train_data)
    data = mean_impute(data)


if __name__ == '__main__':
    args = process_config(sys.argv[1])
    train(args)
