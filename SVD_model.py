import numpy as np
from utils import _convert_df_to_matrix, preprocess, compute_rmse, generate_submission
import os

class SVD_model:
    def __init__(self, args):
        self.args = args
        self.rank = args.svd_args.rank
        self.imputation = args.svd_args.imputation
        self.save_full_pred = args.cv_args.save_full_pred
        self.cv_model_name = args.cv_args.cv_model_name
        self.data_ensemble_folder = args.ens_args.data_ensemble_folder

    def train(self, df_train):
        print("Start training SVD model ...")
        data_train, _ = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.imputation)

        U, sigma_vec, VT = np.linalg.svd(data_train, full_matrices=False)
        Sigma = np.zeros((1000, 1000))
        Sigma[:self.rank, :self.rank] = np.diag(sigma_vec[:self.rank])
        self.reconstructed_matrix = U @ Sigma @ VT * std_train + mean_train

        print("SVD model training ends. ")

    def predict(self, df_test, pred_file_name=None):
        if self.args.generate_submissions:
            submission_file = self.args.submission_folder + "/svd.csv"
            generate_submission(self.args.sample_data, submission_file, self.reconstructed_matrix)
        else:
            predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
            if self.save_full_pred:
                np.savetxt(os.path.join('.', self.data_ensemble_folder, pred_file_name), predictions)
            else:
                labels = df_test['Prediction'].values
                print('RMSE on testing set: {:.4f}'.format(compute_rmse(predictions, labels)))
            return predictions
        return None
