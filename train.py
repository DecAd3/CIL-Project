import sys
import yaml
import argparse
from utils import _read_df_in_format, _convert_df_to_matrix, preprocess, postprocess
from ALS import ALS
from sklearn.model_selection import train_test_split

from utils import _read_df_in_format
from SVD_model import SVD_model


def process_config(path):
    with open(path, 'r') as file:
        yaml_str = file.read()

    data = yaml.safe_load(yaml_str)
    args = argparse.Namespace()

    # Training arguments
    training_args = data['args']['training_args']
    args.train_data = training_args['train_data']
    args.test_size = training_args['test_size']
    args.test_save_dir = training_args['test_save_dir']
    args.model_load_path = training_args['model_load_path']
    args.model_save_path = training_args['model_save_path']
    args.random_seed = training_args['random_seed']
    args.device = training_args['device']
    args.normalization = training_args['normalization']
    args.imputation = training_args['imputation']

    # Experiment arguments
    experiment_args = data['args']['experiment_args']
    args.model_name = experiment_args['model_name']
    args.save_predictions = experiment_args['save_predictions']
    args.predictions_folder = experiment_args['predictions_folder']

    # SVD arguments
    svd_args = data['args']['svd_args']
    args.svd_rank = svd_args['svd_rank']

    # XXX arguments

    return args


def train(args):
    df = _read_df_in_format(args.train_data)
    data, is_provided = _convert_df_to_matrix(df, 10000, 1000)
    data = preprocess(data, 10000, 1000)
    trainer = ALS(data.shape[0], data.shape[1], 10, 10, 0.01)
    trainer.fit(data, is_provided)
    result = trainer.predict()
    result = postprocess(result)
    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.random_seed)

    if args.model_name == 'svd':
        model = SVD_model(args)
    elif args.model_name == 'svd++':
        pass
    elif args.model_name == 'isvd':
        pass
    elif args.model_name == 'als':
        pass

    model.train(df_train)
    model.predict(df_test)


if __name__ == '__main__':
    config_path = 'D:/ETH/2023HS-CIL/CIL-Project/config.yaml'
    args = process_config(config_path)
    # args = process_config(sys.argv[1])
    train(args)
