import numpy as np
from utils import _load_data_for_surprise, _convert_df_to_matrix
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
        self.out_path = './output/svdpp.csv'

    def fit(self, df_train):
        data_train, _ = _convert_df_to_matrix(df_train, 10000, 1000)
        data = _load_data_for_surprise(data_train)

        trainset = data.build_full_trainset()

        # create SVDPP algorithm and train it
        algorithm = SVDpp(n_factors=12, lr_all=0.085, n_epochs=50, reg_all=0.01, verbose=True)
        algorithm.fit(trainset)

        self.algo = algorithm

    def predict(self, df_test):
        with open(self.out_path, 'w+') as f:
            f.write('Id,Prediction\n')
            test = df_test
            for id in test['Id']:
                row, col = id.split('_')
                row = int(row[1:])
                col = int(col[1:])
                uid = row-1
                iid = col-1
                pred = self.algo.predict(uid, iid, verbose=False)
                f.write('r{0}_c{1},{2}\n'.format(row, col, pred))
