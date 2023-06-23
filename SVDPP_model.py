import numpy as np
from utils import _load_data_for_surprise, _convert_df_to_matrix, compute_rmse
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

    def train(self, df_train):
        data = _load_data_for_surprise(df_train)

        trainset = data.build_full_trainset()

        # create SVDPP algorithm and train it
        algorithm = SVDpp(n_factors=9, lr_all=0.008, n_epochs=50, reg_all=0.03, verbose=True, random_state=self.seed)
        algorithm.fit(trainset)
        print(' finish training')
        self.algo = algorithm

    def predict(self, df_test):
        predictions = []
        with open(self.out_path, 'w+') as f:
            f.write('Id,Prediction\n')
            for i in range(df_test.shape[0]):
                row, col = int(df_test['row'].values[i]), int(df_test['col'].values[i])
                # row = int(row[1:])
                # col = int(col[1:])
                uid = row-1
                iid = col-1
                pred = self.algo.predict(uid, iid, verbose=False)
                predictions.append(pred.est)
                f.write('r{0}_c{1},{2}\n'.format(row, col, pred.est))
        labels = df_test['Prediction'].values
        print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))