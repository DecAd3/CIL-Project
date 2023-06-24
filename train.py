import sys
import yaml
import argparse
from ALS_model import ALS_model
from sklearn.model_selection import train_test_split

from utils import _read_df_in_format
from SVD_model import SVD_model
from SVDPP_model import SVDPP_model
from ISVD_model import ISVD_model


def process_config(path):
    with open(path, 'r') as file:
        yaml_str = file.read()

    data = yaml.safe_load(yaml_str)
    args = argparse.Namespace()

    # Training arguments
    training_args = data['args']['training_args']
    args.train_data = training_args['train_data']
    args.sample_data = training_args['sample_data']
    args.num_users = training_args['num_users']
    args.num_items = training_args['num_items']
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
    args.generate_submissions = experiment_args['generate_submissions']
    args.submission_folder = experiment_args['submission_folder']
    args.verbose = experiment_args['verbose']

    # SVD arguments
    svd_args = data['args']['svd_args']
    args.svd_args = argparse.Namespace()
    args.rank = svd_args['rank']

    # ALS arguments
    als_args = data['args']['als_args']
    args.als_args = argparse.Namespace()
    args.als_args.svd_rank = als_args['svd_rank']
    args.als_args.num_iterations = als_args['num_iterations']
    args.als_args.reg_param = als_args['reg_param']
    args.als_args.latent_dim = als_args['latent_dim']
    args.als_args.imputation = als_args['imputation']
    
    # ISVD arguments
    isvd_args = data['args']['isvd_args']
    args.isvd_args = argparse.Namespace()
    args.num_iterations = isvd_args['num_iterations']
    args.type = isvd_args['type']
    args.eta = isvd_args['eta']
    args.rank = isvd_args['rank']
    args.shrinkage = isvd_args['shrinkage']

    # SVDPP arguments
    svdpp_args = data['args']['svdpp_args']
    args.svdpp_args = argparse.Namespace()
    args.svdpp_args.n_factors = svdpp_args['n_factors']
    args.svdpp_args.lr_all = svdpp_args['lr_all']
    args.svdpp_args.n_epochs = svdpp_args['n_epochs']
    args.svdpp_args.reg_all = svdpp_args['reg_all']

    return args


def train(args):

    df = _read_df_in_format(args.train_data)
    if not args.generate_submissions:
        df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.random_seed)
    else:
        df_train, df_test = df, None

    if args.model_name == 'svd':
        model = SVD_model(args)
    elif args.model_name == 'svd++':
        model = SVDPP_model(args)
    elif args.model_name == 'isvd':
        model = ISVD_model(args)
    elif args.model_name == 'als':
        model = ALS_model(args)

    model.train(df_train)
    model.predict(df_test)


if __name__ == '__main__':
    args = process_config(sys.argv[1])
    train(args)