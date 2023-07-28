from ISVD_model import ISVD_model
from ALS_model import ALS_model
import numpy as np
from utils import _convert_df_to_matrix, preprocess, postprocess, compute_rmse, generate_submission

class ISVD_ALS_model:
    def __init__(self, args):
        self.ISVD = ISVD_model(args)
        self.ALS = ALS_model(args)
        
    def train(self, df_train):
        self.ISVD.train(df_train)
        initialization = self.ISVD.obtain_U_VT_as_initialization()
        self.ALS.train(df_train, initialization)
    
    def predict(self, df_test):
        return self.ALS.predict(df_test)