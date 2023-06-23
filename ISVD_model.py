import numpy as np
import os
from utils import _convert_df_to_matrix, preprocess, compute_rmse


class ISVD_model:
    def __init__(self, args):
        self.args = args

    def train(self, df_train):
        print("Start training Iterative-SVD model ...")
        data_train, mask_train = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.args.imputation)

        if self.args.isvd_type == "svp":
            print("Iterative-SVD model: Singular Value Projection. ")
            At = data_train.copy()
            for i in range(self.args.isvd_iter):
                At += self.args.isvd_eta * mask_train * (data_train - At)
                U, sigma_vec, VT = np.linalg.svd(At, full_matrices=False)
                Sigma = np.zeros((1000, 1000))
                rank = self.args.isvd_rank
                Sigma[:rank, :rank] = np.diag(sigma_vec[:rank])
                At = (U @ Sigma @ VT).clip(min=1, max=5)
                print("Iteration {}/{} finished. ".format(i + 1, self.args.isvd_iter))
            self.reconstructed_matrix = At

        if self.args.isvd_type == "nnr":
            print("Iterative-SVD model: Nuclear Norm Relaxation. ")
            At = data_train.copy()
            for i in range(self.args.isvd_iter):
                U, sigma_vec, VT = np.linalg.svd(At, full_matrices=False)
                Sigma = np.diag((sigma_vec - self.args.shrinkage).clip(min=0))
                At = U @ Sigma @ VT
                At[mask_train] = data_train[mask_train]
                At = At.clip(min=1, max=5)
                print("Iteration {}/{} finished. ".format(i+1, self.args.isvd_iter))
            self.reconstructed_matrix = At

        print("Training ends. ")

    def predict(self, df_test):
        predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
        labels = df_test['Prediction'].values
        print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))

        save_predictions = self.args.save_predictions
        if save_predictions:
            np.savetxt(os.path.join('.', self.args.predictions_folder, self.args.model_name + '_pred_full.txt'), self.reconstructed_matrix)
            np.savetxt(os.path.join('.', self.args.predictions_folder, self.args.model_name + '_pred_test.txt'), predictions)

