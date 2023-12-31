import sys
import yaml
import argparse
import os
from utils import _read_df_in_format
from sklearn.model_selection import train_test_split

from SVD_model import SVD_model
from SVDPP_model import SVDPP_model
from ISVD_model import ISVD_model
from ALS_model import ALS_model
from ISVD_ALS_model import ISVD_ALS_model
from BFM_model import BFM_model
from neural import NCF_model
from VAE_model import VAE_model
from ensemble import Ensemble_Model
import time


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
    args.random_seed = training_args['random_seed']
    args.device = training_args['device']
    args.min_rate = training_args['min_rate']
    args.max_rate = training_args['max_rate']

    # Experiment arguments
    experiment_args = data['args']['experiment_args']
    args.model_name = experiment_args['model_name']
    args.model_instance_name = experiment_args['model_instance_name']
    args.generate_submissions = experiment_args['generate_submissions']
    args.submission_folder = experiment_args['submission_folder']
    args.save_full_pred = experiment_args['save_full_pred']
    args.verbose = experiment_args['verbose']

    # SVD arguments
    svd_args = data['args']['svd_args']
    args.svd_args = argparse.Namespace()
    args.svd_args.rank = svd_args['rank']
    args.svd_args.imputation = svd_args['imputation']

    # ALS arguments
    als_args = data['args']['als_args']
    args.als_args = argparse.Namespace()
    # args.als_args.svd_rank = als_args['svd_rank']
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

    # VAE arguments
    vae_args = data['args']['vae_args']
    args.vae_args = argparse.Namespace()
    args.vae_args.num_iterations = vae_args['num_iterations']
    args.vae_args.batch_size = vae_args['batch_size']
    args.vae_args.hidden_dim = vae_args['hidden_dim']
    args.vae_args.latent_dim = vae_args['latent_dim']
    args.vae_args.dropout = vae_args['dropout']
    args.vae_args.lr = vae_args['lr']
    args.vae_args.weight_decay = vae_args['weight_decay']
    args.vae_args.gamma = vae_args['gamma']
    args.vae_args.beta = vae_args['beta']

    # ensemble arguments
    ens_args = data['args']['ensemble_args']
    args.ens_args = argparse.Namespace()
    args.ens_args.fold_number = ens_args['fold_number']
    args.ens_args.shuffle = ens_args['shuffle']
    args.ens_args.regressor = ens_args['regressor']
    args.ens_args.data_ensemble_folder = ens_args['data_ensemble']
    args.ens_args.model_list = ens_args['models']

    # cross validation arguments
    cv_args = data['args']['cv_args']
    args.cv_args = argparse.Namespace()
    args.cv_args.save_full_pred = args.save_full_pred
    args.cv_args.fold_number = args.ens_args.fold_number
    args.cv_args.cv_folder = args.ens_args.data_ensemble_folder
    args.cv_args.cv_model_name = args.model_instance_name
    args.cv_args.weight_entries = cv_args['weight_entries']
    args.cv_args.full_pred_provided = cv_args['full_pred_provided']
    args.cv_args.full_pred_fn = cv_args['full_pred_fn']
    args.cv_args.sample_proportion = cv_args['sample_proportion']
    return args


def get_model(args, df_train = None, df_test = None):
    if args.model_name == 'svd':
        return SVD_model(args)
    elif args.model_name == 'svdpp':
        return SVDPP_model(args)
    elif args.model_name == 'isvd':
        return ISVD_model(args)
    elif args.model_name == 'als':
        return ALS_model(args)
    elif args.model_name == 'isvd+als':
        return ISVD_ALS_model(args)
    elif args.model_name == 'bfm':
        return BFM_model(args)
    elif args.model_name == 'ncf':
        return NCF_model(args)
    elif args.model_name == 'vae':
        return VAE_model(args, df_train, df_test)
    elif args.model_name == 'ensemble':
        return Ensemble_Model(args)

def train(args):
    df = _read_df_in_format(args.train_data)
    if args.generate_submissions or args.model_name == 'ensemble':
        df_train, df_test = df, None
    else:
        df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.random_seed)

    model = get_model(args, df_train, df_test)    

    start_time = time.time()
    model.train(df_train)   # , (df_test, test_blocks)
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time
    model.predict(df_test)
    print("training time: " + str(execution_time) + " seconds")
    
if __name__ == '__main__':
    args = process_config(sys.argv[1])

    if not os.path.exists(args.submission_folder):
        os.makedirs(args.submission_folder)

    train(args)