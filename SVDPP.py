import numpy as np
from utils import _load_data_for_surprise
import os
from surprise import SVDpp
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
import pandas as pd

class SVDPP:
    def __init__(self, args):
        '''
        Model for Singular Value Decomposition Plus Plus(SVDPP)
        '''
        self.test_path = args.test_data
        self.algo = None
        self.out_path = './output/svdpp.csv'

    def fit(self, data_matrix):
        data = _load_data_for_surprise(data_matrix)

        trainset = data.build_full_trainset()

        # create SVDPP algorithm and train it
        algorithm = SVDpp(n_factors=12, lr_all=0.085, n_epochs=50, reg_all=0.01, verbose=True)
        algorithm.fit(trainset)

        self.algo = algorithm

    def predict(self):
        with open(self.out_path, 'w+') as f:
            f.write('Id,Prediction\n')
            test = pd.read_csv(self.test_path)
            for id in test['Id']:
                row, col = id.split('_')
                row = int(row[1:])
                col = int(col[1:])
                uid = row-1
                iid = col-1
                pred = self.algo.predict(uid, iid, verbose=False)
                f.write('r{0}_c{1},{2}\n'.format(row, col, pred))
