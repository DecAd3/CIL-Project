import numpy as np
from utils import _convert_df_to_matrix, preprocess, compute_rmse, generate_submission


class ISVD_model:
    def __init__(self, args):
        self.args = args
        self.type = args.isvd_args.type
        self.imputation = args.isvd_args.imputation
        self.num_iterations = args.isvd_args.num_iterations
        self.eta = args.isvd_args.eta
        self.rank = args.isvd_args.rank
        self.shrinkage = args.isvd_args.shrinkage

    def train(self, df_train):
        print("Start training Iterative-SVD model ...")
        data_train, mask_train = _convert_df_to_matrix(df_train, 10000, 1000)
        data_train, mean_train, std_train = preprocess(data_train, 10000, 1000, self.imputation)

        if self.type == "svp":
            print("Iterative-SVD model: Singular Value Projection. ")
            At = data_train.copy()
            for i in range(self.num_iterations):
                At += self.eta * mask_train * (data_train - At)
                U, sigma_vec, VT = np.linalg.svd(At, full_matrices=False)
                Sigma = np.zeros((1000, 1000))
                rank = self.rank
                Sigma[:rank, :rank] = np.diag(sigma_vec[:rank])
                At = (U @ Sigma @ VT).clip(min=1, max=5)
                print("Iteration {}/{} finished. ".format(i + 1, self.num_iterations))
            self.reconstructed_matrix = At

        if self.type == "nnr":
            print("Iterative-SVD model: Nuclear Norm Relaxation. ")
            At = data_train.copy()
            for i in range(self.num_iterations):
                U, sigma_vec, VT = np.linalg.svd(At, full_matrices=False)
                Sigma = np.diag((sigma_vec - self.shrinkage).clip(min=0))
                At = U @ Sigma @ VT
                At[mask_train] = data_train[mask_train]
                At = At.clip(min=1, max=5)
                print("Iteration {}/{} finished. ".format(i+1, self.num_iterations))
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

    def obtain_U_VT_as_initialization(self):
        U, sigma_vec, VT = np.linalg.svd(self.reconstructed_matrix)
        return [U, VT]