import numpy as np
import os
from utils import _convert_df_to_matrix, preprocess, compute_rmse


class SVD_model:
    def __init__(self, args):
        self.model_name = args.model_name
        self.imputation = args.imputation
        self.rank = args.svd_rank
        self.predictions_folder = args.predictions_folder

    def train(self, df_train, args):
        data_train, _ = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.imputation)

        U, sigma_vec, VT = np.linalg.svd(data_train, full_matrices=False)
        Sigma = np.zeros((data_train.shape[1], data_train.shape[1]))
        Sigma[:self.rank, :self.rank] = np.diag(sigma_vec[:self.rank])
        self.reconstructed_matrix = U @ Sigma @ VT * std_train + mean_train

    def predict(self, df_test, args):
        predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
        ratings = df_test['Prediction'].values
        print('Validation loss: {:.4f}'.format(compute_rmse(predictions, ratings)))

        save_predictions = args.save_predictions
        if save_predictions:
            np.savetxt(os.path.join('.', self.predictions_folder, self.model_name + '_pred_full.txt'), self.reconstructed_matrix)
            np.savetxt(os.path.join('.', self.predictions_folder, self.model_name + '_pred_test.txt'), predictions)

