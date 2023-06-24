import numpy as np
from utils import _convert_df_to_matrix, preprocess, compute_rmse, generate_submission


class ISVD_model:
    def __init__(self, args):
        self.args = args

    def train(self, df_train):
        print("Start training Iterative-SVD model ...")
        data_train, mask_train = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.args.imputation)

        if self.args.type == "svp":
            print("Iterative-SVD model: Singular Value Projection. ")
            At = data_train.copy()
            for i in range(self.args.num_iterations):
                At += self.args.eta * mask_train * (data_train - At)
                U, sigma_vec, VT = np.linalg.svd(At, full_matrices=False)
                Sigma = np.zeros((1000, 1000))
                rank = self.args.rank
                Sigma[:rank, :rank] = np.diag(sigma_vec[:rank])
                At = (U @ Sigma @ VT).clip(min=1, max=5)
                print("Iteration {}/{} finished. ".format(i + 1, self.args.num_iterations))
            self.reconstructed_matrix = At

        if self.args.type == "nnr":
            print("Iterative-SVD model: Nuclear Norm Relaxation. ")
            At = data_train.copy()
            for i in range(self.args.num_iterations):
                U, sigma_vec, VT = np.linalg.svd(At, full_matrices=False)
                Sigma = np.diag((sigma_vec - self.args.shrinkage).clip(min=0))
                At = U @ Sigma @ VT
                At[mask_train] = data_train[mask_train]
                At = At.clip(min=1, max=5)
                print("Iteration {}/{} finished. ".format(i+1, self.args.num_iterations))
            self.reconstructed_matrix = At

        print("Training ends. ")

    def predict(self, df_test):
        if not self.args.generate_submissions:
            predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
            labels = df_test['Prediction'].values
            print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))

        else:
            submission_file = self.args.submission_folder + "/isvd.csv"
            generate_submission(self.args.sample_data, submission_file, self.reconstructed_matrix)