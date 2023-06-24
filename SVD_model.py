import numpy as np
import os
from utils import _convert_df_to_matrix, preprocess, compute_rmse, generate_submission


class SVD_model:
    def __init__(self, args):
        self.args = args

    def train(self, df_train):
        print("Start training SVD model ...")
        data_train, _ = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.args.imputation)

        U, sigma_vec, VT = np.linalg.svd(data_train, full_matrices=False)
        Sigma = np.zeros((1000, 1000))
        rank = self.args.svd_rank
        Sigma[:rank, :rank] = np.diag(sigma_vec[:rank])
        self.reconstructed_matrix = U @ Sigma @ VT * std_train + mean_train

        print("Training ends. ")

    def predict(self, df_test):
        if not self.args.generate_submissions:
            predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
            labels = df_test['Prediction'].values
            print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))

        else:
            submission_file = self.args.submission_folder + "/svd.csv"
            generate_submission(self.args.sample_data, submission_file, self.reconstructed_matrix)

