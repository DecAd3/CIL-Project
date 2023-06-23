import numpy as np
import os
from utils import _convert_df_to_matrix, preprocess, compute_rmse


class ISVD_model:
    def __init__(self, args):
        self.args = args

    def train(self, df_train):
        data_train, _ = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mask_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.args.imputation)

        X = data_train.copy()
        for i in range(self.args.num_iter):
            U, sigma_vec, VT = np.linalg.svd(X, full_matrices=False)
            Sigma = np.diag((sigma_vec - self.args.shrinkage).clip(min=0))
            X = U @ Sigma @ VT
            X[mask_train] = data_train[mask_train]
            X = X.clip(min=1, max=5)
            print("Iteration {}/{} finished. ".format(i+1, self.args.num_iter))
        self.reconstructed_matrix = X

    def predict(self, df_test):
        predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
        labels = df_test['Prediction'].values
        print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))

        save_predictions = self.args.save_predictions
        if save_predictions:
            np.savetxt(os.path.join('.', self.args.predictions_folder, self.args.model_name + '_pred_full.txt'), self.reconstructed_matrix)
            np.savetxt(os.path.join('.', self.args.predictions_folder, self.args.model_name + '_pred_test.txt'), predictions)

