import sys
import yaml
import argparse
from utils import _read_df_in_format
from sklearn.model_selection import train_test_split

from SVD_model import SVD_model
from SVDPP_model import SVDPP_model
from ISVD_model import ISVD_model
from ALS_model import ALS_model
from ISVD_ALS_model import ISVD_ALS_model
from BFM_model import BFM_model
from neural import NCF_model


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
    args.min_rate = training_args['min_rate']
    args.max_rate = training_args['max_rate']

    # Experiment arguments
    experiment_args = data['args']['experiment_args']
    args.model_name = experiment_args['model_name']
    args.generate_submissions = experiment_args['generate_submissions']
    args.submission_folder = experiment_args['submission_folder']
    args.verbose = experiment_args['verbose']

    # SVD arguments
    svd_args = data['args']['svd_args']
    args.svd_args = argparse.Namespace()
    args.svd_args.rank = svd_args['rank']

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
    args.isvd_args.num_iterations = isvd_args['num_iterations']
    args.isvd_args.imputation = isvd_args['imputation']
    args.isvd_args.type = isvd_args['type']
    args.isvd_args.eta = isvd_args['eta']
    args.isvd_args.rank = isvd_args['rank']
    args.isvd_args.shrinkage = isvd_args['shrinkage']

    # SVDPP arguments
    svdpp_args = data['args']['svdpp_args']
    args.svdpp_args = argparse.Namespace()
    args.svdpp_args.n_factors = svdpp_args['n_factors']
    args.svdpp_args.lr_all = svdpp_args['lr_all']
    args.svdpp_args.n_epochs = svdpp_args['n_epochs']
    args.svdpp_args.reg_all = svdpp_args['reg_all']

    # BFM arguments
    bfm_args = data['args']['bfm_args']
    args.bfm_args = argparse.Namespace()
    args.bfm_args.algorithm = bfm_args['algorithm']
    args.bfm_args.iteration = bfm_args['iteration']
    args.bfm_args.dimension = bfm_args['dimension']
    args.bfm_args.use_iu = bfm_args['use_iu']
    args.bfm_args.use_ii = bfm_args['use_ii']
    args.bfm_args.variational = bfm_args['variational']

    # NCF arguments
    ncf_args = data['args']['ncf_args']
    args.ncf_args = argparse.Namespace()
    args.ncf_args.EPOCHS = ncf_args['EPOCHS']
    args.ncf_args.BATCH_SIZE = ncf_args['BATCH_SIZE']
    args.ncf_args.n_factors = ncf_args['n_factors']
    args.ncf_args.learning_rate = ncf_args['learning_rate']
    args.ncf_args.train_file = ncf_args['train_file']
    args.ncf_args.save_file = ncf_args['save_file']
    args.ncf_args.all_predictions_file = ncf_args['all_predictions_file']

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
    elif args.model_name == 'isvd+als':
        model = ISVD_ALS_model(args)
    elif args.model_name == 'bfm':
        model = BFM_model(args)
    elif args.model_name == 'ncf':
        model = NCF_model(args)

    model.train(df_train)
    model.predict(df_test)


if __name__ == '__main__':
    args = process_config(sys.argv[1])
    train(args)