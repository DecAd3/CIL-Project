from train import process_config, get_model
import os 
import sys
import numpy as np
from utils import _convert_df_to_matrix, preprocess, postprocess, compute_rmse, generate_submission, _read_df_in_format
from sklearn.model_selection import KFold
import random
from cross_validation import cross_validation
from itertools import product
import pandas as pd


def grid_search(args):
    # An example of param_grid
    # param_grid = {
    # 'C': [0.1, 1, 10, 100],
    # 'gamma': [0.001, 0.01, 0.1, 1],
    # 'kernel': ['linear', 'rbf']
    # }
    param_grid = {
        'rank': [7, 8, 9, 10]
    }

    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    df = pd.DataFrame(param_combinations)
    df['RMSE'] = np.zeros(len(param_combinations))

    best_params = param_combinations[0]
    best_rmse = np.Infinity

    for i, params in enumerate(param_combinations):
        print("Performing cross validation on params: {}".format(params))
        # change parameters here, if needed
        if args.model_name == 'svd':
            args.svd_args.rank = params['rank']
        elif args.model_name == 'svdpp':
            args.svdpp_args.n_factors = params['n_factors']
            args.svdpp_args.lr_all = params['lr_all']
            args.svdpp_args.n_epochs = params['n_epochs']
            args.svdpp_args.reg_all = params['reg_all']
        elif args.model_name == 'isvd':
            args.isvd_args.num_iterations = params['num_iterations']
            args.isvd_args.imputation = params['imputation']
            args.isvd_args.type = params['type']
            args.isvd_args.eta = params['eta']
            args.isvd_args.rank = params['rank']
            args.isvd_args.shrinkage = params['shrinkage']
        elif args.model_name == 'als':
            args.als_args.svd_rank = params['svd_rank']
            args.als_args.num_iterations = params['num_iterations']
            args.als_args.reg_param = params['reg_param']
        elif args.model_name == 'bfm':
            args.bfm_args.iteration = params['iteration']
            args.bfm_args.dimension = params['dimension']
        elif args.model_name == 'vae':
            args.vae_args.num_iterations = params['num_iterations']
            args.vae_args.dropout = params['dropout']
            args.vae_args.beta = params['beta']
        rmse = cross_validation(args)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
        df.iloc[i, -1] = rmse
    df.to_csv('./output/{}_{}_fold_grid_search.csv'.format(args.model_name, args.cv_args.fold_number))
    print('The best params are: {}'.format(best_params))
    print('The best rmse is {} on {} fold cross-validation'.format(best_rmse, args.cv_args.fold_number))

if __name__ == '__main__':
    args = process_config(sys.argv[1])
    grid_search(args)
    
