import numpy as np
from utils import _load_data_for_surprise, _read_df_in_format, compute_rmse
import os
from surprise import SVDpp
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
import pandas as pd

class SVDPP_model:
    def __init__(self, args):
        '''
        Model for Singular Value Decomposition Plus Plus(SVDPP)
        '''
        self.algo = None
        self.seed = args.random_seed
        self.out_path = './output/svdpp.csv'
        self.verbose = args.verbose
        self.n_factors = args.svdpp_args.n_factors
        self.lr_all = args.svdpp_args.lr_all
        self.n_epochs = args.svdpp_args.n_epochs
        self.reg_all = args.svdpp_args.reg_all
        self.generate_submissions = args.generate_submissions
        self.sample_data = args.sample_data

    def train(self, df_train):
        data = _load_data_for_surprise(df_train)

        trainset = data.build_full_trainset()

        # create SVDPP algorithm and train it
        algorithm = SVDpp(n_factors=self.n_factors, lr_all=self.lr_all, n_epochs=self.n_epochs,
                           reg_all=self.reg_all, verbose=self.verbose, random_state=self.seed)
        algorithm.fit(trainset)
        print(' finish training')
        self.algo = algorithm

    def predict(self, df_test):
        if self.generate_submissions:
            df_test = _read_df_in_format(self.sample_data)
        predictions = []
        with open(self.out_path, 'w+') as f:
            f.write('Id,Prediction\n')
            for i in range(df_test.shape[0]):
                row, col = int(df_test['row'].values[i]), int(df_test['col'].values[i])
                uid = row-1
                iid = col-1
                pred = self.algo.predict(uid, iid, verbose=False)
                predictions.append(pred.est)
                f.write('r{0}_c{1},{2}\n'.format(row, col, pred.est))
        if not self.generate_submissions:
            labels = df_test['Prediction'].values
            print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))